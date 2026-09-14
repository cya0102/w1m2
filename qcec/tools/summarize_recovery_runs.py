"""Summarize validation-selected ActivityNet recovery runs.

Run labels should encode the variant and seed, for example
``B_s8=checkpoints/...``.  The script selects each run's checkpoint by
Validation ``R@1,mIoU`` and never reads Test metrics.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    ('metrics', 'R@1,mIoU', 'r1_mIoU'),
    ('metrics', 'R@1,IoU@0.5', 'r1_iou05'),
    ('metrics', 'R@5,mIoU', 'r5_mIoU'),
    ('metrics', 'R@5,IoU@0.5', 'r5_iou05'),
    ('diagnostics', 'gt_short_R5_IoU@0.5', 'short_r5_iou05'),
)


def parse_run_spec(value):
    if '=' not in value:
        raise ValueError('--run must have LABEL=RUN_DIRECTORY form')
    label, directory = value.split('=', 1)
    if not label or not directory:
        raise ValueError('--run contains an empty label or directory')
    return label, Path(directory)


def _load_run(label, directory):
    metadata_path = directory / 'run_metadata.json'
    metrics_path = directory / 'metrics.jsonl'
    if not metrics_path.exists():
        raise FileNotFoundError('missing {}'.format(metrics_path))
    rows = []
    with metrics_path.open(encoding='utf8') as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get('kind') == 'eval'
                    and row.get('split') == 'Validation'):
                rows.append(row)
    if not rows:
        raise ValueError('no Validation rows found in {}'.format(metrics_path))
    selected = max(rows, key=lambda row: row['metrics']['R@1,mIoU'])
    values = {'label': label, 'directory': str(directory),
              'epoch': selected.get('epoch')}
    for section, key, output_name in METRICS:
        values[output_name] = selected.get(section, {}).get(key, '')
    if metadata_path.exists():
        with metadata_path.open(encoding='utf8') as handle:
            metadata = json.load(handle)
        values['seed'] = metadata.get('seed', '')
    else:
        values['seed'] = ''
    match = re.match(r'^(?P<variant>[A-Za-z]+).*?(?P<seed>\d+)$', label)
    values['variant'] = match.group('variant') if match else label
    if values['seed'] == '' and match:
        values['seed'] = int(match.group('seed'))
    return values


def summarize(rows):
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row['variant']].append(row)
    summary = {}
    for variant, variant_rows in sorted(by_variant.items()):
        result = {'num_runs': len(variant_rows), 'seeds': [], 'metrics': {}}
        for row in variant_rows:
            if row['seed'] != '':
                result['seeds'].append(row['seed'])
        for _, _, output_name in METRICS:
            numeric = [float(row[output_name]) for row in variant_rows
                       if row[output_name] != '']
            if not numeric:
                continue
            result['metrics'][output_name] = {
                'mean': float(np.mean(numeric)),
                'std': float(np.std(numeric)),
                'values': numeric,
            }
        summary[variant] = result
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True,
                        help='LABEL=run_directory; repeat for each seed/variant')
    parser.add_argument('--output', default='diagnostics/activitynet/recovery_summary.json')
    args = parser.parse_args()
    rows = [_load_run(*parse_run_spec(value)) for value in args.run]
    payload = {'runs': rows, 'summary': summarize(rows),
               'selection_rule': 'Validation R@1,mIoU'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    csv_path = output.with_suffix('.csv')
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('saved {} and {}'.format(output, csv_path))


if __name__ == '__main__':
    main()
