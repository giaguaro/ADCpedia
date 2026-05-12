import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import RDKit_DESCRIPTOR_COLUMNS, SPECIFIC_GENES


def _prefix_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    def key(c):
        m = re.search(r'_(\d+)$', c)
        return int(m.group(1)) if m else c
    return sorted(cols, key=key)


class ADCPredictionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.f3_cols = _prefix_cols(self.df, 'Gene_PCA_esm_embeddings_')
        self.f4_cols = _prefix_cols(self.df, 'NN_mRNA_Cell_Line_Embedding_')
        self.f5_cols = _prefix_cols(self.df, 'NN_Gene_mRNA_Counts_Embedding_')
        self.maccs_cols = _prefix_cols(self.df, 'MACCS_')
        self.f9_cols = [f'{g}_mRNA_count_scaled' for g in SPECIFIC_GENES]
        self.f10_cols = [f'Predicted_Protein_Intensity_{g}' for g in SPECIFIC_GENES]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_1 = torch.tensor(row[[
            'Drug_Antibody_Ratio_(DAR)',
            'Is_Tubulin_Target',
            'Is_DNA_Target',
            'Is_not_Tubulin_DNA_Target',
        ]].values.astype(np.float32))
        return {
            'feature_1': feature_1,
            'feature_2': torch.tensor([row['Predicted_Protein_Intensity']], dtype=torch.float32),
            'feature_3': torch.tensor(row[self.f3_cols].values.astype(np.float32)),
            'feature_4': torch.tensor(row[self.f4_cols].values.astype(np.float32)),
            'feature_5': torch.tensor(row[self.f5_cols].values.astype(np.float32)),
            'feature_6': torch.tensor(row[RDKit_DESCRIPTOR_COLUMNS].values.astype(np.float32)),
            'feature_7': torch.tensor(row[self.maccs_cols].values.astype(np.float32)),
            'feature_9': torch.tensor(row[self.f9_cols].values.astype(np.float32)),
            'feature_10': torch.tensor(row[self.f10_cols].values.astype(np.float32)),
        }


class ADCTrainingDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data.fillna(0, inplace=True)
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data.fillna(0, inplace=True)
        self.pred_ds = ADCPredictionDataset(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item = self.pred_ds[idx]
        item['flag'] = torch.tensor(row['flag'], dtype=torch.float32)
        item['ic50'] = torch.tensor(min(float(row['InVitro_Equalized_IC50_Value']), 1e9), dtype=torch.float32)
        return item
