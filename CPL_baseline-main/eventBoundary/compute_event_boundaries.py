#!/usr/bin/env python3
"""Compute event boundaries for Charades-STA and ActivityNet Captions.

The script discovers the CPL annotation and feature paths from the existing
dataset configs, de-duplicates annotation rows by video id, and writes one
compact JSON record per video.  It can also consume any compatible HDF5 file
through ``--feature-path``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    from .event_boundary import (
        CONTRASTIVE_KERNEL,
        detect_event_boundaries,
        resample_features_like_cpl,
    )
except ImportError:  # Support ``python eventBoundary/compute_event_boundaries.py``.
    from event_boundary import (
        CONTRASTIVE_KERNEL,
        detect_event_boundaries,
        resample_features_like_cpl,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_TO_CONFIG_KEY = {
    "train": "train_data",
    "val": "val_data",
    "test": "test_data",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_path: Path
    hdf5_feature_key: Optional[str]
    clip_feature_candidates: Tuple[Path, ...]


DATASET_SPECS = {
    "charades": DatasetSpec(
        name="charades",
        config_path=PROJECT_ROOT / "config" / "charades" / "main.json",
        hdf5_feature_key=None,
        clip_feature_candidates=(
            Path(
                "/data/chenyuan/videogrounding/VGDataset/Charades-STA/clip/"
                "clip_vit_b16_video_features.hdf5"
            ),
        ),
    ),
    "activitynet": DatasetSpec(
        name="activitynet",
        config_path=PROJECT_ROOT / "config" / "activitynet" / "main.json",
        hdf5_feature_key="c3d_features",
        clip_feature_candidates=(
            Path(
                "/data/chenyuan/videogrounding/VGDataset/ActivityNet/clip/"
                "clip_vit_b16_video_features.hdf5"
            ),
        ),
    ),
}


def _load_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError("JSON file does not exist: {}".format(path))


def _load_config(path: Path) -> Mapping[str, object]:
    config = _load_json(path)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("config must contain a 'dataset' object: {}".format(path))
    return config


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _annotation_paths(
    config: Mapping[str, object], splits: Sequence[str]
) -> List[Path]:
    dataset_config = config["dataset"]
    assert isinstance(dataset_config, dict)
    paths: List[Path] = []
    seen = set()
    for split in splits:
        key = SPLIT_TO_CONFIG_KEY[split]
        value = dataset_config.get(key)
        if not isinstance(value, str):
            raise ValueError("dataset config is missing string field '{}'".format(key))
        path = _resolve_project_path(value).resolve()
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _read_video_durations(annotation_paths: Sequence[Path]) -> "OrderedDict[str, float]":
    """Read the CPL list format and preserve deterministic first-seen order."""

    durations: "OrderedDict[str, float]" = OrderedDict()
    for annotation_path in annotation_paths:
        rows = _load_json(annotation_path)
        if not isinstance(rows, list):
            raise ValueError("annotation root must be a list: {}".format(annotation_path))
        for row_index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) < 2:
                raise ValueError(
                    "invalid annotation row {} in {}".format(row_index, annotation_path)
                )
            video_id = str(row[0])
            duration = float(row[1])
            if not np.isfinite(duration) or duration < 0:
                raise ValueError(
                    "invalid duration for video '{}' in {}".format(video_id, annotation_path)
                )
            previous = durations.get(video_id)
            if previous is not None and not np.isclose(previous, duration, atol=1e-6):
                raise ValueError(
                    "conflicting durations for video '{}': {} and {}".format(
                        video_id, previous, duration
                    )
                )
            durations.setdefault(video_id, duration)
    return durations


def _feature_candidates(
    spec: DatasetSpec,
    config: Mapping[str, object],
    feature_source: str,
    explicit_path: Optional[Path],
) -> List[Path]:
    if explicit_path is not None:
        return [explicit_path.expanduser()]

    if feature_source == "clip":
        return list(spec.clip_feature_candidates)

    dataset_config = config["dataset"]
    assert isinstance(dataset_config, dict)
    candidates: List[Path] = []
    configured = dataset_config.get("feature_path")
    if isinstance(configured, str):
        candidates.append(_resolve_project_path(configured))

    if spec.name == "charades":
        candidates.extend(
            [
                PROJECT_ROOT / "data" / "charades" / "i3d_features.hdf5",
            ]
        )
    else:
        candidates.extend(
            [
                PROJECT_ROOT
                / "data"
                / "activitynet"
                / "sub_activitynet_v1-3.c3d.hdf5",
                PROJECT_ROOT / "data" / "activitynet" / "c3d_features.hdf5",
            ]
        )
    return candidates


def _resolve_feature_path(candidates: Sequence[Path]) -> Path:
    attempted = []
    for candidate in candidates:
        resolved = candidate.resolve()
        attempted.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "could not find a feature HDF5 file; tried:\n  - {}".format(
            "\n  - ".join(attempted)
        )
    )


def _read_hdf5_features(
    handle: h5py.File,
    video_id: str,
    feature_key: Optional[str],
    strict_feature_key: bool = False,
) -> np.ndarray:
    if video_id not in handle:
        raise KeyError("video '{}' is missing from the feature HDF5".format(video_id))
    node = handle[video_id]
    if isinstance(node, h5py.Dataset):
        if strict_feature_key and feature_key is not None:
            raise KeyError(
                "--feature-key '{}' was provided, but video '{}' is a root-level "
                "dataset rather than a group".format(feature_key, video_id)
            )
        return np.asarray(node, dtype=np.float32)
    if not isinstance(node, h5py.Group):
        raise TypeError("unsupported HDF5 object for video '{}'".format(video_id))

    if feature_key is not None and feature_key in node:
        dataset = node[feature_key]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(
                "HDF5 key '{}' for video '{}' is not a dataset".format(
                    feature_key, video_id
                )
            )
    elif strict_feature_key and feature_key is not None:
        raise KeyError(
            "requested --feature-key '{}' is missing for video '{}'; group keys "
            "are {}".format(feature_key, video_id, list(node.keys()))
        )
    elif "c3d_features" in node:
        dataset = node["c3d_features"]
    else:
        dataset_names = [key for key, value in node.items() if isinstance(value, h5py.Dataset)]
        if len(dataset_names) != 1:
            raise KeyError(
                "cannot choose a feature dataset for video '{}'; group keys are {}".format(
                    video_id, list(node.keys())
                )
            )
        dataset = node[dataset_names[0]]
    return np.asarray(dataset, dtype=np.float32)


def _default_output_path(
    dataset: str, feature_source: str, native_length: bool
) -> Path:
    qualifiers = []
    if feature_source != "config":
        qualifiers.append(feature_source)
    if native_length:
        qualifiers.append("native")
    middle = "_" + "_".join(qualifiers) if qualifiers else ""
    return PROJECT_ROOT / "eventBoundary" / "outputs" / (
        dataset + middle + "_event_boundaries.json"
    )


def _atomic_write_json(payload: Mapping[str, object], output_path: Path, pretty: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(output_path.parent),
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
            handle.write("\n")
        os.chmod(temporary_name, 0o664)
        os.replace(temporary_name, output_path)
    except BaseException:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise


def compute_dataset(
    dataset: str,
    config_path: Optional[Path] = None,
    feature_source: str = "config",
    feature_path: Optional[Path] = None,
    feature_key: Optional[str] = None,
    annotation_paths: Optional[Sequence[Path]] = None,
    splits: Sequence[str] = ("train", "val", "test"),
    num_clips: Optional[int] = None,
    native_length: bool = False,
    all_feature_videos: bool = False,
    include_all_scores: bool = False,
    limit: Optional[int] = None,
    progress_every: int = 250,
    quiet: bool = False,
) -> Tuple[Dict[str, object], Path]:
    """Compute one dataset and return ``(JSON payload, resolved feature path)``."""

    if dataset not in DATASET_SPECS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    if feature_source not in ("config", "clip"):
        raise ValueError("feature_source must be 'config' or 'clip'")
    if native_length and num_clips is not None:
        raise ValueError("native_length and num_clips are mutually exclusive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    spec = DATASET_SPECS[dataset]
    resolved_config_path = (config_path or spec.config_path).expanduser().resolve()
    config = _load_config(resolved_config_path)
    dataset_config = config["dataset"]
    assert isinstance(dataset_config, dict)

    if annotation_paths is None:
        resolved_annotation_paths = _annotation_paths(config, splits)
    else:
        resolved_annotation_paths = [path.expanduser().resolve() for path in annotation_paths]
    durations = _read_video_durations(resolved_annotation_paths)

    resolved_feature_path = _resolve_feature_path(
        _feature_candidates(spec, config, feature_source, feature_path)
    )
    effective_feature_key = feature_key
    if effective_feature_key is None and feature_source == "config":
        effective_feature_key = spec.hdf5_feature_key

    effective_num_clips: Optional[int]
    if native_length:
        effective_num_clips = None
    elif num_clips is not None:
        if num_clips <= 0:
            raise ValueError("num_clips must be positive")
        effective_num_clips = int(num_clips)
    else:
        configured_num_clips = dataset_config.get("max_num_frames")
        if not isinstance(configured_num_clips, int) or configured_num_clips <= 0:
            raise ValueError("dataset config must define a positive max_num_frames")
        effective_num_clips = configured_num_clips

    start_time = time.monotonic()
    videos: "OrderedDict[str, object]" = OrderedDict()
    with h5py.File(str(resolved_feature_path), "r") as feature_file:
        if all_feature_videos:
            video_ids = sorted(str(video_id) for video_id in feature_file.keys())
        else:
            video_ids = sorted(durations.keys())
        if limit is not None:
            video_ids = video_ids[:limit]

        missing = [video_id for video_id in video_ids if video_id not in feature_file]
        if missing:
            raise KeyError(
                "{} requested videos are missing from {}; examples: {}".format(
                    len(missing), resolved_feature_path, missing[:10]
                )
            )

        total = len(video_ids)
        for index, video_id in enumerate(video_ids, start=1):
            source_features = _read_hdf5_features(
                feature_file,
                video_id,
                effective_feature_key,
                strict_feature_key=feature_key is not None,
            )
            source_num_clips = int(source_features.shape[0])
            features = (
                source_features
                if effective_num_clips is None
                else resample_features_like_cpl(source_features, effective_num_clips)
            )
            result = detect_event_boundaries(features)
            videos[video_id] = result.to_record(
                source_num_clips=source_num_clips,
                duration=durations.get(video_id),
                include_all_scores=include_all_scores,
            )

            if (
                not quiet
                and (index == total or index == 1 or index % progress_every == 0)
            ):
                elapsed = time.monotonic() - start_time
                rate = index / elapsed if elapsed > 0 else 0.0
                print(
                    "[{}] {}/{} videos ({:.1f} videos/s)".format(
                        dataset, index, total, rate
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        hdf5_attributes = {
            str(key): _json_safe_attribute(value)
            for key, value in feature_file.attrs.items()
        }

    payload: Dict[str, object] = {
        "metadata": {
            "schema_version": 1,
            "dataset": dataset,
            "paper_section": "III-E.1, Eq. (6)",
            "feature_source": feature_source,
            "feature_path": str(resolved_feature_path),
            "feature_key": effective_feature_key,
            "feature_hdf5_attributes": hdf5_attributes,
            "config_path": str(resolved_config_path),
            "annotation_paths": [str(path) for path in resolved_annotation_paths],
            "splits": list(splits),
            "all_feature_videos": bool(all_feature_videos),
            "limit": limit,
            "resample_num_clips": effective_num_clips,
            "video_count": len(videos),
            "contrastive_kernel": CONTRASTIVE_KERNEL.astype(int).tolist(),
            "threshold": "per-video mean of raw boundary scores",
            "max_filter_size": 3,
            "local_max_tie_policy": "retain all equal maxima (cited-reference behavior)",
            "padding": "zero",
            "endpoint_policy": "force clip indices 0 and num_clips-1 as delimiters",
            "boundary_index_convention": "zero-based clip-center index",
            "segment_index_convention": "half-open [start, end), with cuts 0 and num_clips",
            "normalized_event_span_convention": (
                "event_spans_normalized_cw uses consecutive reference boundary "
                "coordinates divided by num_clips; pooling_event_spans_normalized_cw "
                "uses full-coverage half-open segments"
            ),
            "time_mapping": "boundary_index / (num_clips-1) * video_duration",
            "segment_time_mapping": "segment_cut / num_clips * video_duration",
            "include_all_scores": bool(include_all_scores),
        },
        "videos": videos,
    }
    return payload, resolved_feature_path


def _json_safe_attribute(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_safe_attribute(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paper-faithful TSM contrastive-kernel event boundaries "
            "for the CPL Charades/ActivityNet data."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("charades", "activitynet", "all"),
        default="all",
        help="dataset to process (default: both)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="override config path (only valid when processing one dataset)",
    )
    parser.add_argument(
        "--feature-source",
        choices=("config", "clip"),
        default="config",
        help=(
            "use the CPL config feature_path, or the locally available CLIP "
            "ViT-B/16 HDF5 (default: config)"
        ),
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        help="explicit HDF5 path; overrides --feature-source discovery",
    )
    parser.add_argument(
        "--feature-key",
        help="dataset name inside each per-video HDF5 group (auto-detected by default)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        help="explicit CPL-format annotation JSON files",
    )
    parser.add_argument(
        "--splits",
        choices=("train", "val", "test"),
        nargs="+",
        default=("train", "val", "test"),
        help="annotation splits used to select videos (default: all)",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--num-clips",
        type=int,
        help="CPL-style uniform resampling length (default: config max_num_frames)",
    )
    sampling_group.add_argument(
        "--native-length",
        action="store_true",
        help="detect boundaries on every HDF5 clip without CPL resampling",
    )
    parser.add_argument(
        "--all-feature-videos",
        action="store_true",
        help="process every HDF5 root key, not only annotation video ids",
    )
    parser.add_argument(
        "--include-all-scores",
        action="store_true",
        help="store the dense raw score vector for every video (larger JSON)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON for one dataset, or output directory with --dataset all",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    parser.add_argument("--limit", type=int, help="process only the first N videos")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="progress interval in videos (default: 250)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.dataset == "all" and (
        args.config is not None
        or args.feature_path is not None
        or args.feature_key is not None
        or args.annotations is not None
    ):
        parser.error(
            "--config, --feature-path, --feature-key, and --annotations require "
            "a single --dataset"
        )
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.num_clips is not None and args.num_clips <= 0:
        parser.error("--num-clips must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    datasets = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    output_override = args.output.expanduser().resolve() if args.output else None
    if args.dataset == "all" and output_override is not None and output_override.suffix:
        parser.error("--output must be a directory when --dataset all is used")

    output_paths = {}
    for dataset in datasets:
        if output_override is None:
            output_path = _default_output_path(
                dataset, args.feature_source, args.native_length
            )
        elif args.dataset == "all":
            output_path = output_override / _default_output_path(
                dataset, args.feature_source, args.native_length
            ).name
        else:
            output_path = output_override
        output_paths[dataset] = output_path
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                "output already exists (pass --overwrite to replace it): {}".format(
                    output_path
                )
            )

    for dataset in datasets:
        output_path = output_paths[dataset]
        payload, _ = compute_dataset(
            dataset=dataset,
            config_path=args.config,
            feature_source=args.feature_source,
            feature_path=args.feature_path,
            feature_key=args.feature_key,
            annotation_paths=args.annotations,
            splits=args.splits,
            num_clips=args.num_clips,
            native_length=args.native_length,
            all_feature_videos=args.all_feature_videos,
            include_all_scores=args.include_all_scores,
            limit=args.limit,
            progress_every=args.progress_every,
            quiet=args.quiet,
        )
        _atomic_write_json(payload, output_path, pretty=args.pretty)
        if not args.quiet:
            print("wrote {}".format(output_path), file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
