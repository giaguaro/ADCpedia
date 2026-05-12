import logging

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl


class ProteinIntensityModel(pl.LightningModule):
    def __init__(self, esm_embedding_dim, mrna_embedding_dim, read_count_dim=1,
                 output_dim=1, hidden_dim=512, reduced_hidden_dim=64, dropout_prob=0.15):
        super().__init__()
        self.esm_layer = nn.Sequential(
            nn.Linear(esm_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
        )
        self.mrna_layer = nn.Sequential(
            nn.Linear(mrna_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
        )
        self.read_count_layer = nn.Sequential(
            nn.Linear(read_count_dim, reduced_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(reduced_hidden_dim),
            nn.Dropout(dropout_prob),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + reduced_hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, esm_embeddings, mrna_embeddings, read_count):
        if esm_embeddings.dim() == 1:
            esm_embeddings = esm_embeddings.unsqueeze(0)
        if mrna_embeddings.dim() == 1:
            mrna_embeddings = mrna_embeddings.unsqueeze(0)
        if read_count.dim() == 1:
            read_count = read_count.unsqueeze(1)
        x_esm = self.esm_layer(esm_embeddings.float())
        x_mrna = self.mrna_layer(mrna_embeddings.float())
        x_rc = self.read_count_layer(read_count.float())
        x = torch.cat((x_esm, x_mrna, x_rc), dim=1)
        out = self.regressor(x)
        return out, x_esm, x_mrna, x_rc


def load_esm_model(model_name, device):
    import esm
    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter()


def compute_esm_embedding(seq, model, batch_converter, alphabet, device, repr_layer=33):
    if not seq or len(seq) < 10:
        return None
    data = [('0', seq)]
    _, _, toks = batch_converter(data)
    toks = toks.to(device)
    lens = (toks != alphabet.padding_idx).sum(1)
    try:
        with torch.no_grad():
            out = model(toks, repr_layers=[repr_layer])
        reps = out['representations'][repr_layer]
        return reps[0, 1:lens[0]-1].mean(0).cpu().numpy()
    except Exception as e:
        logging.error('ESM failed: %s', e)
        return None


def esm_pca_for_sequence(seq, esm_model, batch_converter, alphabet, device, scaler, pca, repr_layer=33):
    emb = compute_esm_embedding(seq, esm_model, batch_converter, alphabet, device, repr_layer)
    if emb is None:
        return None
    emb = emb.astype(np.float32)
    emb_sc = scaler.transform([emb])
    return pca.transform(emb_sc)[0].astype(np.float32)
