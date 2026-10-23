"""Protocol demonstration only. Copy as ml_backend.py in a TEST checkout.

This deliberately does no training and must never be used for competition scoring.
"""
import csv
import json
from pathlib import Path
import sys


def main():
    request_path, response_path = map(Path, sys.argv[1:])
    envelope = json.loads(request_path.read_text(encoding='utf-8'))
    if envelope['version'] != 1:
        raise ValueError('unsupported protocol')
    task_dir = Path(envelope['task_dir'])
    output = task_dir / 'mock_candidate.csv'
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'prediction'])
        writer.writerows([['a', 1.0], ['b', 2.0]])
    response_path.write_text(json.dumps({'version': 1, 'status': 'ok',
        'candidate_path': output.name, 'schema': {'columns': ['id', 'prediction'],
        'expected_rows': 2, 'id_columns': ['id'], 'numeric_columns': ['prediction']},
        'notes': 'Synthetic integration fixture; no ML score.'}), encoding='utf-8')


if __name__ == '__main__':
    main()
