#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


import argparse

from amm_adc.pi_table import run_pi_table
from amm_adc.settings import load_config


def main():
    ap = argparse.ArgumentParser(description='Predict and merge protein intensities')
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--validation-csv', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--gene-sequences-csv', default=None)
    ap.add_argument('--batch-size', type=int, default=1024)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_pi_table(args, cfg)


if __name__ == '__main__':
    main()
