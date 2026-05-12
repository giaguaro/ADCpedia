from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .constants import SPECIFIC_GENES
from .protein import ProteinIntensityModel, load_esm_model, esm_pca_for_sequence
from .rnaseq import load_transcriptomic_data, prepare_rnaseq, name_resolver, resolve_name, cell_line_pca
from .settings import repo_path


TARGET_ANTIGENS = ['CD19', 'CD22', 'ERBB2', 'NECTIN4', 'TACSTD2']


def _norm(s):
    return ''.join(ch for ch in str(s).upper() if ch.isalnum())


ANTIGEN_SYNONYMS = {
    _norm('HER2'): 'ERBB2',
    _norm('Nectin-4'): 'NECTIN4',
    _norm('NECTIN-4'): 'NECTIN4',
    _norm('Trop2'): 'TACSTD2',
    _norm('TROP2'): 'TACSTD2',
    _norm('TROP-2'): 'TACSTD2',
}

CELL_SYNONYMS = {
    _norm('AU 565'): 'AU565',
    _norm('H226'): 'NCI-H226',
    _norm('NCIH226'): 'NCI-H226',
    _norm('SKOV3'): 'SK-OV-3',
    _norm('SK-OV3'): 'SK-OV-3',
    _norm('SKOV-3'): 'SK-OV-3',
}


def canonical_antigen(x):
    return ANTIGEN_SYNONYMS.get(_norm(x), str(x).upper())


def canonical_cellline(x):
    return CELL_SYNONYMS.get(_norm(x), str(x))


class PIDataset(Dataset):
    def __init__(self, df, esm_cols, mrna_cols):
        self.esm = df[esm_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        self.mrna = df[mrna_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        self.rc = pd.to_numeric(df['read_count'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.rc)

    def __getitem__(self, idx):
        return torch.from_numpy(self.esm[idx]), torch.from_numpy(self.mrna[idx]), torch.from_numpy(self.rc[idx])


def run_pi_table(args, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device') != 'cpu' else 'cpu')
    val = pd.read_csv(args.validation_csv)
    if 'CellLine' not in val.columns or 'Antigen' not in val.columns:
        raise ValueError('validation CSV must include CellLine and Antigen')
    val['CellLine'] = val['CellLine'].map(canonical_cellline)
    val['Antigen'] = val['Antigen'].map(canonical_antigen)

    genes_needed = sorted(set(SPECIFIC_GENES + TARGET_ANTIGENS))
    cell_lines_needed = sorted(val['CellLine'].dropna().unique().tolist())

    rnaseq, model_name_to_index = load_transcriptomic_data(repo_path(cfg, cfg['paths']['transcriptomic_hdf5']))
    symbol_to_row, expr_mat = prepare_rnaseq(rnaseq)
    resolver = name_resolver(model_name_to_index)
    cl_to_hdf5 = {cl: resolve_name(cl, resolver) for cl in cell_lines_needed}
    missing = [cl for cl, key in cl_to_hdf5.items() if key is None]
    if missing:
        raise ValueError(f'cell lines not found in HDF5: {missing}')

    scaler_prot = joblib.load(repo_path(cfg, cfg['paths']['scaler_proteomics']))
    pca_prot = joblib.load(repo_path(cfg, cfg['paths']['pca_proteomics']))
    scaler_tx = joblib.load(repo_path(cfg, cfg['paths']['scaler_transcriptomics']))
    pca_tx = joblib.load(repo_path(cfg, cfg['paths']['pca_transcriptomics']))
    scaler_rc = joblib.load(repo_path(cfg, cfg['paths']['scaler_read_count']))

    seqs = {}
    if args.gene_sequences_csv:
        gdf = pd.read_csv(args.gene_sequences_csv)
    else:
        gdf = pd.read_csv(repo_path(cfg, cfg['paths']['specific_genes_file']))
    if 'Gene' in gdf.columns and 'Sequence' in gdf.columns:
        for _, r in gdf.iterrows():
            g = str(r['Gene']).strip().upper()
            s = r['Sequence']
            if isinstance(s, str) and len(s) > 10:
                seqs[g] = s

    esm_model, alphabet, batch_converter = load_esm_model(cfg['esm']['model_name'], device)
    repr_layer = int(cfg['esm'].get('repr_layer', 33))

    gene_to_esm = {}
    missing_seq = []
    for g in tqdm(genes_needed, desc='ESM PCA', unit='gene'):
        seq = seqs.get(g.upper())
        if not seq:
            missing_seq.append(g)
            continue
        emb = esm_pca_for_sequence(seq, esm_model, batch_converter, alphabet, device, scaler_prot, pca_prot, repr_layer)
        if emb is None:
            missing_seq.append(g)
        else:
            gene_to_esm[g.upper()] = emb
    if missing_seq:
        raise ValueError(f'missing sequences for: {missing_seq}')

    cell_to_pca = {}
    for cl, hkey in cl_to_hdf5.items():
        cell_to_pca[cl] = cell_line_pca(hkey, model_name_to_index, expr_mat, scaler_tx, pca_tx)

    rows = []
    esm_cols = [f'PCA_esm_embeddings_{i}' for i in range(pca_prot.n_components_)]
    mrna_cols = [f'PCA_mRNA_Component_{i}' for i in range(pca_tx.n_components_)]
    for cl in cell_lines_needed:
        hkey = cl_to_hdf5[cl]
        cidx = model_name_to_index[hkey]
        for g in genes_needed:
            ridx = symbol_to_row.get(g.upper())
            if ridx is None or g.upper() not in gene_to_esm:
                continue
            rc_raw = float(expr_mat[ridx, cidx])
            rc_scaled = float(scaler_rc.transform([[rc_raw]]).ravel()[0])
            d = {'CellLine': cl, 'Gene': g.upper(), 'read_count': rc_scaled}
            ev = gene_to_esm[g.upper()]
            mv = cell_to_pca[cl]
            for i, v in enumerate(ev):
                d[esm_cols[i]] = float(v)
            for i, v in enumerate(mv):
                d[mrna_cols[i]] = float(v)
            rows.append(d)
    feat = pd.DataFrame(rows)

    model = ProteinIntensityModel.load_from_checkpoint(
        str(repo_path(cfg, cfg['paths']['protein_intensity_checkpoint'])),
        esm_embedding_dim=pca_prot.n_components_,
        mrna_embedding_dim=pca_tx.n_components_,
        read_count_dim=1,
        strict=False,
    ).to(device)
    model.eval()

    ds = PIDataset(feat, esm_cols, mrna_cols)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    preds = []
    for esm_x, mrna_x, rc_x in tqdm(dl, desc='Protein intensity', unit='batch'):
        with torch.no_grad():
            out, _, _, _ = model(esm_x.to(device), mrna_x.to(device), rc_x.to(device))
        preds.append(out.squeeze(1).cpu().numpy())
    feat['Predicted_Protein_Intensity'] = np.concatenate(preds).astype(np.float32)

    pred_df = feat[['CellLine', 'Gene', 'Predicted_Protein_Intensity']].groupby(['CellLine', 'Gene'], as_index=False).mean()
    antigen_pred = pred_df[pred_df['Gene'].isin(TARGET_ANTIGENS)].rename(columns={'Gene': 'Antigen'})
    out = val.merge(antigen_pred, on=['CellLine', 'Antigen'], how='left')
    resist = pred_df[pred_df['Gene'].isin(SPECIFIC_GENES)].pivot_table(index='CellLine', columns='Gene', values='Predicted_Protein_Intensity', aggfunc='mean').reset_index()
    resist = resist.rename(columns={g: f'Predicted_Protein_Intensity_{g}' for g in resist.columns if g != 'CellLine'})
    out = out.merge(resist, on='CellLine', how='left')
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    return out
