"""Merge per-row chunk pickles produced by run_sensitivity.py into per-pair
matrix pickles that are drop-in replacements for `data/sensitivity_<p1>_<p2>.pkl`.

Usage:
    python savio_jobs/merge_chunks.py \
        --in-dir  /global/scratch/users/<user>/raas_sensitivity_<DATE>/chunks \
        --out-dir data

The output schema matches the original notebook:
    {'matrix': np.ndarray (n1, n2),
     'axis1_name': str, 'axis1_values': np.ndarray,
     'axis2_name': str, 'axis2_values': np.ndarray}
"""
import argparse
import os
import pickle
from collections import defaultdict
from itertools import combinations

import numpy as np


PARAM_NAMES = ['R', 'F', 'h', 'lam']
PAIRS = list(combinations(PARAM_NAMES, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', required=True,
                        help='Directory containing sensitivity_<p1>_<p2>_row<NNN>.pkl')
    parser.add_argument('--out-dir', required=True,
                        help='Where to write per-pair matrix pickles')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Write a pair even if some rows are missing (NaN-fill them)')
    args = parser.parse_args()

    rows_by_pair = defaultdict(dict)

    for fname in sorted(os.listdir(args.in_dir)):
        if not fname.startswith('sensitivity_') or not fname.endswith('.pkl'):
            continue
        path = os.path.join(args.in_dir, fname)
        with open(path, 'rb') as f:
            d = pickle.load(f)
        pair = tuple(d['pair'])
        rows_by_pair[pair][int(d['row_idx'])] = d

    if not rows_by_pair:
        raise SystemExit(f'No row pickles found in {args.in_dir}')

    os.makedirs(args.out_dir, exist_ok=True)

    expected_pairs = set(PAIRS)
    found_pairs = set(rows_by_pair.keys())
    missing_pairs = expected_pairs - found_pairs
    if missing_pairs:
        msg = f'Missing pairs entirely: {sorted(missing_pairs)}'
        if not args.allow_incomplete:
            raise SystemExit(msg + '  (re-run with --allow-incomplete to skip)')
        print('WARN:', msg)

    for pair, rows in rows_by_pair.items():
        p1, p2 = pair
        n_grid = max(r['n_grid'] for r in rows.values())

        present = sorted(rows)
        missing = sorted(set(range(n_grid)) - set(present))
        if missing:
            msg = (f'pair ({p1}, {p2}): missing rows {missing} of {n_grid}')
            if not args.allow_incomplete:
                raise SystemExit('ERROR: ' + msg + '  (rerun those array tasks)')
            print('WARN:', msg)

        any_row = rows[present[0]]
        n_cols = len(any_row['p2_values'])
        matrix = np.full((n_grid, n_cols), np.nan)
        p1_values = np.full(n_grid, np.nan)
        p2_values = any_row['p2_values']
        for i in present:
            r = rows[i]
            matrix[i, :] = r['row_thresholds']
            p1_values[i] = r['p1_value']

        out = {
            'matrix':       matrix,
            'axis1_name':   p1, 'axis1_values': p1_values,
            'axis2_name':   p2, 'axis2_values': p2_values,
            'n_grid':       n_grid,
            'vi_iters':     any_row.get('vi_iters'),
            'vi_samples':   any_row.get('vi_samples'),
        }
        out_path = os.path.join(args.out_dir, f'sensitivity_{p1}_{p2}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)

        n_nan = int(np.isnan(matrix).sum())
        print(f'wrote {out_path}: shape {matrix.shape}, '
              f'{n_nan} NaN cells, {len(present)}/{n_grid} rows present')


if __name__ == '__main__':
    main()
