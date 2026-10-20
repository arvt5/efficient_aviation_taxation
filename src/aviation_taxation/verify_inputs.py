"""Verify the byte integrity of distributed archived inputs and configurations."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    records = json.loads((root / 'provenance/files.json').read_text())
    failed = []
    for record in records:
        path = root / record['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failed.append(record['path'])
    if failed:
        raise SystemExit('Input integrity check failed: ' + ', '.join(failed))
    print(f'Verified {len(records)} distributed input hashes')


if __name__ == '__main__':
    main()
