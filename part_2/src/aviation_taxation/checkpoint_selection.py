from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

def read_json(path: Path) -> dict:
    return json.loads(path.read_text())

def direct_metric(path: Path) -> float:
    data = read_json(path)
    return float(data['direct_forward_optimization_top']['max_direct_gain_scaled_relative_to_baseline'])

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--direct-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    candidates = []
    for policy in sorted(args.checkpoint_dir.glob('outer_*_policy.npz')):
        tag = policy.name.removesuffix('_policy.npz')
        summary = args.direct_root / tag / 'summary.json'
        if not summary.is_file():
            continue
        candidates.append({'tag': tag, 'policy': str(policy), 'summary': str(summary), 'direct_max_relative': direct_metric(summary)})
    if not candidates:
        raise RuntimeError('No directly certified accepted checkpoints found')
    best = min(candidates, key=lambda row: row['direct_max_relative'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best['policy'], args.out_dir / 'best_policy.npz')
    shutil.copy2(best['summary'], args.out_dir / 'best_summary.json')
    result = {'best': best, 'candidates': candidates}
    (args.out_dir / 'selection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
