from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from eventBoundary.compute_event_boundaries import compute_dataset
from eventBoundary.event_boundary import (
    CONTRASTIVE_KERNEL,
    compute_boundary_scores,
    detect_event_boundaries,
    resample_features_like_cpl,
    temporal_self_similarity,
)


class EventBoundaryTest(unittest.TestCase):
    def test_kernel_matches_paper_equation(self):
        expected = np.asarray(
            [
                [1, 1, 0, -1, -1],
                [1, 1, 0, -1, -1],
                [0, 0, 0, 0, 0],
                [-1, -1, 0, 1, 1],
                [-1, -1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(CONTRASTIVE_KERNEL, expected)

    def test_fast_scores_equal_explicit_tsm_convolution(self):
        rng = np.random.RandomState(17)
        for num_clips in (1, 2, 4, 5, 11, 29):
            features = rng.randn(num_clips, 13).astype(np.float32)
            tsm = temporal_self_similarity(features)
            padded = np.pad(tsm, ((2, 2), (2, 2)), mode="constant")
            explicit = np.asarray(
                [
                    np.sum(
                        padded[index : index + 5, index : index + 5]
                        * CONTRASTIVE_KERNEL,
                        dtype=np.float32,
                    )
                    for index in range(num_clips)
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                compute_boundary_scores(features), explicit, rtol=2e-5, atol=2e-5
            )

    def test_two_visual_regimes_produce_transition_boundaries(self):
        features = np.vstack(
            [
                np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (8, 1)),
                np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (8, 1)),
            ]
        )
        result = detect_event_boundaries(features)

        self.assertEqual(result.boundary_indices[0], 0)
        self.assertEqual(result.boundary_indices[-1], len(features) - 1)
        # A hard step produces a two-clip plateau around the transition.  The
        # size-3 max filter retains ties, exactly as the reference code does.
        self.assertIn(7, result.detected_boundary_indices())
        self.assertIn(8, result.detected_boundary_indices())

    def test_half_open_events_cover_every_clip_once(self):
        rng = np.random.RandomState(3)
        result = detect_event_boundaries(rng.randn(23, 7).astype(np.float32))
        intervals = result.event_intervals_indices_half_open()

        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], 23)
        for left, right in zip(intervals[:-1], intervals[1:]):
            self.assertEqual(left[1], right[0])
        self.assertEqual(sum(end - start for start, end in intervals), 23)
        self.assertAlmostEqual(
            sum(span[1] for span in result.event_spans_half_open_normalized_cw()),
            1.0,
        )

    def test_single_clip_has_one_full_coverage_pooling_event(self):
        result = detect_event_boundaries(np.asarray([[1.0, 2.0]], dtype=np.float32))
        np.testing.assert_array_equal(result.boundary_indices, [0])
        self.assertEqual(result.detected_boundary_indices(), [])
        self.assertEqual(result.event_spans_normalized_cw(), [])
        self.assertEqual(result.event_intervals_indices_half_open(), [[0, 1]])
        self.assertEqual(result.event_spans_half_open_normalized_cw(), [[0.5, 1.0]])

    def test_cpl_resampling_preserves_baseline_rounding(self):
        features = np.arange(5, dtype=np.float32).reshape(5, 1)
        sampled = resample_features_like_cpl(features, num_clips=3)
        np.testing.assert_allclose(
            sampled[:, 0], np.asarray([0.5, 2.0, 3.0], dtype=np.float32)
        )

    def test_dataset_processor_supports_both_hdf5_layouts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps([["video_a", 12.0, [1.0, 4.0], "example"]]),
                encoding="utf-8",
            )

            for dataset, grouped in (("charades", False), ("activitynet", True)):
                config_path = root / (dataset + "_config.json")
                feature_path = root / (dataset + "_features.hdf5")
                config_path.write_text(
                    json.dumps(
                        {
                            "dataset": {
                                "feature_path": str(feature_path),
                                "max_num_frames": 6,
                                "train_data": str(annotation_path),
                                "val_data": str(annotation_path),
                                "test_data": str(annotation_path),
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                with h5py.File(str(feature_path), "w") as handle:
                    values = np.arange(32, dtype=np.float32).reshape(8, 4)
                    if grouped:
                        group = handle.create_group("video_a")
                        group.create_dataset("c3d_features", data=values)
                    else:
                        handle.create_dataset("video_a", data=values)

                payload, resolved_path = compute_dataset(
                    dataset=dataset,
                    config_path=config_path,
                    feature_path=feature_path,
                    annotation_paths=[annotation_path],
                    quiet=True,
                )
                self.assertEqual(resolved_path, feature_path.resolve())
                self.assertEqual(payload["metadata"]["video_count"], 1)
                record = payload["videos"]["video_a"]
                self.assertEqual(record["source_num_clips"], 8)
                self.assertEqual(record["num_clips"], 6)
                self.assertEqual(record["segmentation_cuts_indices"][0], 0)
                self.assertEqual(record["segmentation_cuts_indices"][-1], 6)

                native_payload, _ = compute_dataset(
                    dataset=dataset,
                    config_path=config_path,
                    feature_path=feature_path,
                    annotation_paths=[annotation_path],
                    native_length=True,
                    quiet=True,
                )
                self.assertEqual(native_payload["videos"]["video_a"]["num_clips"], 8)

                if grouped:
                    with self.assertRaisesRegex(KeyError, "missing_feature"):
                        compute_dataset(
                            dataset=dataset,
                            config_path=config_path,
                            feature_path=feature_path,
                            feature_key="missing_feature",
                            annotation_paths=[annotation_path],
                            quiet=True,
                        )


if __name__ == "__main__":
    unittest.main()
