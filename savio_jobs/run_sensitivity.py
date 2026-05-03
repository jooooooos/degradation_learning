"""Compute one row of one parameter-pair's sensitivity heatmap.

Designed for SLURM array job: SLURM_ARRAY_TASK_ID -> (pair_index, row_index).
Each task computes value iteration for `n_grid` cells (one full row of one
pair's matrix) and writes a single pickle to `--out-dir`.

Usage (under SLURM):
    SLURM_ARRAY_TASK_ID=0 python run_sensitivity.py --n-grid 30 --out-dir <dir>

Local test:
    python run_sensitivity.py --n-grid 5 --out-dir /tmp/sens_test --task-id 0
"""
import argparse
import os
import pickle
import sys
import time
from copy import deepcopy
from itertools import combinations

import numpy as np

# Make raas/ importable regardless of where this script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

import raas.config as config                             # noqa: E402
from raas.utils import get_perfect_degradation_learner   # noqa: E402


PARAM_NAMES = ['R', 'F', 'h', 'lam']
PAIRS = list(combinations(PARAM_NAMES, 2))

DEFAULTS = {
    'R':   config.mdp_params['replacement_cost'],
    'F':   config.mdp_params['failure_cost'],
    'h':   config.mdp_params['holding_cost_rate'],
    'lam': config.LAMBDA_VAL,
}


def get_scalar_threshold(dpagent):
    """Median across t of the smallest cc index where action=2 in policy_departure.

    Mirrors notebooks/utils.py exactly so the output is comparable to the
    existing `data/sensitivity_*.pkl` files.
    """
    n_t = len(dpagent.grids[4])
    thresholds = np.full(n_t, np.nan)
    for t_idx in range(n_t):
        col = dpagent.policy_departure[:, t_idx]
        replace_mask = (col == 2)
        if np.any(replace_mask):
            thresholds[t_idx] = float(dpagent.grids[0][np.argmax(replace_mask)])
        else:
            # Sentinel: never replace within grid
            thresholds[t_idx] = float(dpagent.grids[0][-1])
    return float(np.median(thresholds))


def build_params(overrides):
    """Build (mdp_params dict, lambda value) from a dict of parameter overrides."""
    mdp = deepcopy(config.mdp_params)
    lam = config.LAMBDA_VAL
    for name, val in overrides.items():
        if name == 'R':
            mdp['replacement_cost'] = val
        elif name == 'F':
            mdp['failure_cost'] = val
        elif name == 'h':
            mdp['holding_cost_rate'] = val
        elif name == 'lam':
            lam = val
        else:
            raise ValueError(f'Unknown parameter override: {name}')
    return mdp, lam


def make_ranges(n_grid, low_mult=0.25, high_mult=4.0):
    """Same convention as the original notebook: 0.25x to 4x of default."""
    return {
        name: np.linspace(
            low_mult * DEFAULTS[name],
            high_mult * DEFAULTS[name],
            n_grid,
        )
        for name in PARAM_NAMES
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-grid', type=int, default=30,
                        help='Grid points per axis (default: 30; original used 15)')
    parser.add_argument('--out-dir', required=True,
                        help='Directory for per-row pickles')
    parser.add_argument('--vi-iters', type=int, default=200,
                        help='Value iteration max iterations')
    parser.add_argument('--vi-samples', type=int, default=300_000,
                        help='Customer sample size for expectation weights')
    parser.add_argument('--task-id', type=int, default=None,
                        help='Override SLURM_ARRAY_TASK_ID (for local testing)')
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        env_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env_id is None:
            raise SystemExit('SLURM_ARRAY_TASK_ID not set (use --task-id for local runs)')
        task_id = int(env_id)

    N = args.n_grid
    n_pairs = len(PAIRS)
    total_tasks = n_pairs * N
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit(
            f'task_id {task_id} out of range [0, {total_tasks}) for n_grid={N}; '
            f'set --array=0-{total_tasks - 1} in the sbatch script.'
        )

    pair_idx = task_id // N
    row_idx = task_id % N
    p1, p2 = PAIRS[pair_idx]

    ranges = make_ranges(N)
    p1_vals = ranges[p1]
    p2_vals = ranges[p2]
    v1 = float(p1_vals[row_idx])

    print(f'[task {task_id}] pair=({p1}, {p2}) row={row_idx}/{N - 1}  '
          f'{p1}={v1:.6f}  sweep {p2} over {len(p2_vals)} values  '
          f'(n_grid={N}, vi_iters={args.vi_iters}, vi_samples={args.vi_samples})',
          flush=True)

    row_thresholds = np.full(len(p2_vals), np.nan)
    t0 = time.time()
    for j, v2_np in enumerate(p2_vals):
        v2 = float(v2_np)
        mdp, lam = build_params({p1: v1, p2: v2})
        _, dpagent, _ = get_perfect_degradation_learner(
            sample_size=args.vi_samples,
            iterations=args.vi_iters,
            mdp_params=mdp,
            baseline_hazard_lambda=lam,
        )
        row_thresholds[j] = get_scalar_threshold(dpagent)
        print(f'    [{j:3d}/{len(p2_vals) - 1}] {p2}={v2:.6f}  '
              f'c*={row_thresholds[j]:.4f}  '
              f'(elapsed {time.time() - t0:.1f}s)',
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f'sensitivity_{p1}_{p2}_row{row_idx:03d}.pkl'
    out_path = os.path.join(args.out_dir, fname)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'pair': (p1, p2),
            'row_idx': row_idx,
            'p1_name': p1,
            'p2_name': p2,
            'p1_value': v1,
            'p2_values': p2_vals,
            'row_thresholds': row_thresholds,
            'n_grid': N,
            'vi_iters': args.vi_iters,
            'vi_samples': args.vi_samples,
        }, f)
    print(f'[task {task_id}] saved {out_path}  (total {time.time() - t0:.1f}s)',
          flush=True)


if __name__ == '__main__':
    main()
