"""CLI for the complete Practice 7 experiment."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from pathlib import Path
from recommender.experiment import run, MODELS


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Practice 7: MovieLens recommendations and temporal evaluation')
    p.add_argument('--dataset', choices=['movielens'], default='movielens')
    p.add_argument('--data', type=Path, default=root / 'data')
    p.add_argument('--output', type=Path, default=root / 'results')
    p.add_argument('--model', choices=['all'] + MODELS, default='all')
    p.add_argument('--k', type=int, nargs='+', default=[5, 10])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--evaluate', action='store_true', help='Evaluation is always performed; accepted for lecture CLI compatibility')
    args = p.parse_args()
    if any(k < 1 or k > 100 for k in args.k): p.error('K must be in 1..100')
    run(args)

if __name__ == '__main__':
    main()
