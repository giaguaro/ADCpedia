#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


import argparse
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from amm_adc.cnn_model import ADCModel
from amm_adc.datasets import ADCTrainingDataset
from amm_adc.settings import load_config, repo_path


def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser(description='Train the 10 nM ADC CNN')
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))

    params = cfg['cnn_params']
    train_cfg = cfg['training']

    train_ds = ADCTrainingDataset(repo_path(cfg, train_cfg['train_csv']))
    val_ds = ADCTrainingDataset(repo_path(cfg, train_cfg['val_csv']))
    test_ds = ADCTrainingDataset(repo_path(cfg, train_cfg['test_csv']))

    num_workers = int(train_cfg.get('num_workers', 0))
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = ADCModel(
        feature_params=params,
        classifier_params=params,
        lr=params['lr'],
        boundary=float(cfg.get('threshold_nm', 10.0)),
        optimizer_type=params.get('optimizer_type', 'adam'),
        loss=train_cfg.get('loss', 'boundary_weighted_bce'),
    )

    ckpt_dir = repo_path(cfg, train_cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='best_model-10nM-{epoch:02d}-{val_auc:.3f}',
        monitor='val_auc',
        mode='max',
        save_top_k=5,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=int(train_cfg.get('patience', 8)), min_delta=1e-3)
    logger = CSVLogger(save_dir=str(repo_path(cfg, train_cfg.get('logger_dir', 'logs_best_cnn'))), name='best_model_run_10nM')

    trainer = pl.Trainer(
        max_epochs=int(train_cfg.get('max_epochs', 50)),
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        best_model = ADCModel.load_from_checkpoint(best_path, strict=False)
        trainer.test(best_model, test_loader, verbose=True)
    print('Best checkpoint:', best_path)


if __name__ == '__main__':
    main()
