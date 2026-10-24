"""Train and value-refine one equilibrium using an explicit configuration."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .equilibrium import train
from .model import FractionalParams
from .network import FractionalDEQN
from .value_refinement import refine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, help='Override the archived training seed for an independent replication')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.seed is not None:
        if args.seed < 0:
            parser.error('Seed must be nonnegative')
        settings['seed'] = args.seed
        settings['seed_note'] = 'Explicit independent-replication seed'
    params = FractionalParams(**settings['params'])
    torch.manual_seed(settings['seed'])
    net = FractionalDEQN(hidden=settings['hidden'], activation='relu').to(args.device)
    train(net, params, dphi=settings['dt'] / params.d,
          n_iter=settings['iterations'], B=settings['batch_size'],
          lr=settings['learning_rate'], log_every=250,
          two_firm_fraction=0.4 if params.entry_phi > 0 else None,
          cross_weight=1.0, cross_ramp_frac=settings['cross_ramp_fraction'])
    if settings['refinement_rounds']:
        net = refine(net, params, n_outer=settings['refinement_rounds'],
                     inner_iters=settings['refinement_iterations'], w_eval=settings['value_weight'],
                     dphi=settings['dt'] / params.d, lr=max(settings['learning_rate'] * 0.33, 1e-4),
                     batch=settings['batch_size'], value_batch=settings['batch_size'],
                     n_phi=settings['n_phi'], burn_in=settings['burn_in'], device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'params': asdict(params), 'hidden': settings['hidden'], 'activation': 'relu',
                'model_state_dict': net.cpu().state_dict()}, args.output)
    args.output.with_suffix('.settings.json').write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    main()
