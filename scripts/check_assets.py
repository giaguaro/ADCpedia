#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


import argparse
from pathlib import Path

from amm_adc.settings import load_config, repo_path


PREDICTION_KEYS = [
    'transcriptomic_hdf5',
    'scaler_proteomics',
    'pca_proteomics',
    'scaler_transcriptomics',
    'pca_transcriptomics',
    'scaler_read_count',
    'specific_genes_file',
    'protein_intensity_checkpoint',
    'cnn_checkpoint',
]

TRAINING_KEYS = [
    ('training', 'train_csv'),
    ('training', 'val_csv'),
    ('training', 'test_csv'),
]


def main():
    ap = argparse.ArgumentParser(description='Check required AMM ADC files')
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--training', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)

    missing = []
    print('Prediction files')
    for key in PREDICTION_KEYS:
        path = repo_path(cfg, cfg['paths'][key])
        ok = path.exists()
        print(('OK      ' if ok else 'MISSING ') + str(path))
        if not ok:
            missing.append(path)

    opt = repo_path(cfg, cfg['paths'].get('model_list', ''))
    if str(opt):
        print(('OK      ' if opt.exists() else 'OPTION  ') + str(opt))

    if args.training:
        print('\nTraining files')
        for section, key in TRAINING_KEYS:
            path = repo_path(cfg, cfg[section][key])
            ok = path.exists()
            print(('OK      ' if ok else 'MISSING ') + str(path))
            if not ok:
                missing.append(path)

    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
