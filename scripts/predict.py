#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


import argparse

from amm_adc.pipeline import run_prediction
from amm_adc.settings import load_config


def main():
    ap = argparse.ArgumentParser(description='Run 10 nM AMM ADC prediction')
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--input-csv', default=None)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--smiles', default=None)
    ap.add_argument('--cell-line', nargs='*', default=None)
    ap.add_argument('--gene-symbol', nargs='*', default=None)
    ap.add_argument('--dar', type=float, default=3.5)
    ap.add_argument('--is-tubulin', type=int, default=0)
    ap.add_argument('--is-dna', type=int, default=0)
    ap.add_argument('--is-not-tubulin-dna', type=int, default=0)
    ap.add_argument('--uniprot-id', nargs='*', default=None)
    ap.add_argument('--manual-gene-sequence', nargs='*', default=None)
    ap.add_argument('--all-cell-lines', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_prediction(args, cfg)


if __name__ == '__main__':
    main()
