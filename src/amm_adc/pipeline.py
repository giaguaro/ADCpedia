import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .chem import build_chem_table
from .cnn_model import ADCModel
from .constants import RDKit_DESCRIPTOR_COLUMNS, SPECIFIC_GENES
from .datasets import ADCPredictionDataset
from .protein import ProteinIntensityModel, load_esm_model, esm_pca_for_sequence
from .rnaseq import load_transcriptomic_data, prepare_rnaseq, cell_line_pca, counts_table
from .sequences import fill_missing_sequences
from .settings import repo_path


def _device(name):
    if name and name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _path(cfg, key):
    return repo_path(cfg, cfg['paths'][key])


def _input_from_args(args, cfg, model_name_to_index):
    cols = cfg['columns']
    if args.input_csv:
        return pd.read_csv(args.input_csv)

    cell_lines = args.cell_line or []
    if args.all_cell_lines:
        cell_lines = sorted(model_name_to_index.keys())
    genes = args.gene_symbol or []
    if not args.smiles or not genes or (not cell_lines and not args.all_cell_lines):
        raise ValueError('Provide --input-csv or manual --smiles/--gene-symbol/--cell-line arguments')

    seq_map = {}
    if args.manual_gene_sequence and len(args.manual_gene_sequence) == len(genes):
        seq_map = dict(zip(genes, args.manual_gene_sequence))
    uni_map = {}
    if args.uniprot_id and len(args.uniprot_id) == len(genes):
        uni_map = dict(zip(genes, args.uniprot_id))

    rows = []
    for cl in cell_lines:
        for g in genes:
            rows.append({
                cols['smiles']: args.smiles,
                cols['cell_line']: cl,
                cols['gene']: g,
                cols['gene_sequence']: seq_map.get(g, ''),
                cols['dar']: args.dar,
                cols['is_tubulin']: args.is_tubulin,
                cols['is_dna']: args.is_dna,
                cols['is_not_tubulin_dna']: args.is_not_tubulin_dna,
                'UniProt_ID': uni_map.get(g, None),
            })
    return pd.DataFrame(rows)


def _load_specific_gene_sequences(path):
    out = {g: None for g in SPECIFIC_GENES}
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        g = str(r.get('Gene', '')).strip()
        seq = r.get('Sequence', '')
        if g in out and isinstance(seq, str) and len(seq) > 10:
            out[g] = seq
    return out


def _precompute_gene_esm(gene_to_seq, esm_model, batch_converter, alphabet, device, scaler, pca, repr_layer):
    out = {}
    for g, seq in tqdm(gene_to_seq.items(), desc='ESM PCA', unit='gene'):
        if not seq:
            out[g] = np.full((pca.n_components_,), np.nan, dtype=np.float32)
            continue
        emb = esm_pca_for_sequence(seq, esm_model, batch_converter, alphabet, device, scaler, pca, repr_layer)
        if emb is None:
            out[g] = np.full((pca.n_components_,), np.nan, dtype=np.float32)
        else:
            out[g] = emb
    return out


def _predict_pi_batches(model, E, P, R, device, batch_size):
    n = len(R)
    preds = np.empty((n,), dtype=np.float32)
    mrna_parts = []
    rc_parts = []
    for start in tqdm(range(0, n, batch_size), desc='Protein intensity', unit='batch'):
        end = min(start + batch_size, n)
        e_t = torch.tensor(E[start:end], dtype=torch.float32, device=device)
        p_t = torch.tensor(P[start:end], dtype=torch.float32, device=device)
        r_t = torch.tensor(R[start:end], dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            out, x_esm, x_mrna, x_rc = model(e_t, p_t, r_t)
        preds[start:end] = out.squeeze(1).cpu().numpy().astype(np.float32)
        mrna_parts.append(x_mrna.cpu().numpy().astype(np.float32))
        rc_parts.append(x_rc.cpu().numpy().astype(np.float32))
    return preds, np.concatenate(mrna_parts, axis=0), np.concatenate(rc_parts, axis=0)


def run_prediction(args, cfg):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    torch.set_grad_enabled(False)
    device = _device(cfg.get('device'))
    cols = cfg['columns']
    pred_cfg = cfg.get('prediction', {})
    pi_batch_size = int(pred_cfg.get('pi_batch_size', 4096))
    maccs_size = int(cfg.get('molecule', {}).get('maccs_size', 167))
    antibody_mw = float(cfg.get('molecule', {}).get('antibody_mw', 150000.0))

    rnaseq_data, model_name_to_index = load_transcriptomic_data(_path(cfg, 'transcriptomic_hdf5'))
    symbol_to_row, expr_mat = prepare_rnaseq(rnaseq_data)

    scaler_prot = joblib.load(_path(cfg, 'scaler_proteomics'))
    pca_prot = joblib.load(_path(cfg, 'pca_proteomics'))
    scaler_tx = joblib.load(_path(cfg, 'scaler_transcriptomics'))
    pca_tx = joblib.load(_path(cfg, 'pca_transcriptomics'))
    scaler_rc = joblib.load(_path(cfg, 'scaler_read_count'))

    logging.info('Loading ESM model: %s', cfg['esm']['model_name'])
    esm_model, alphabet, batch_converter = load_esm_model(cfg['esm']['model_name'], device)
    repr_layer = int(cfg['esm'].get('repr_layer', 33))

    gene_to_seq = _load_specific_gene_sequences(_path(cfg, 'specific_genes_file'))
    gene2esm = _precompute_gene_esm(gene_to_seq, esm_model, batch_converter, alphabet, device, scaler_prot, pca_prot, repr_layer)

    input_df = _input_from_args(args, cfg, model_name_to_index)
    if args.all_cell_lines and args.input_csv:
        expanded = []
        for _, row in input_df.iterrows():
            base = row.to_dict()
            for cl in sorted(model_name_to_index.keys()):
                d = base.copy()
                d[cols['cell_line']] = cl
                expanded.append(d)
        input_df = pd.DataFrame(expanded)

    input_df = fill_missing_sequences(
        input_df,
        cols['gene'],
        cols['gene_sequence'],
        fetch_missing=bool(pred_cfg.get('fetch_missing_sequences', True)),
    )
    input_df['Main_Gene_Sequence'] = input_df[cols['gene_sequence']]

    unique_smiles = sorted(set(input_df[cols['smiles']].dropna().astype(str)))
    chem_df = build_chem_table(unique_smiles, cols['smiles'], maccs_size)
    input_df = input_df.merge(chem_df, on=cols['smiles'], how='left')
    input_df['adc_mw'] = antibody_mw + input_df[cols['dar']].astype(float) * input_df['payload_mw'].astype(float)

    sym_idx = input_df[cols['gene']].astype(str).map(symbol_to_row)
    cl_idx = input_df[cols['cell_line']].astype(str).map(model_name_to_index)
    antigen_counts = np.full(len(input_df), np.nan, dtype=np.float32)
    mask = sym_idx.notna() & cl_idx.notna()
    if mask.any():
        antigen_counts[mask.to_numpy()] = expr_mat[
            sym_idx[mask].astype(int).to_numpy(),
            cl_idx[mask].astype(int).to_numpy(),
        ]
    input_df['Antigen_Receptor_mRNA_Read_Counts'] = antigen_counts

    all_cls = sorted(model_name_to_index, key=lambda x: model_name_to_index[x])
    cnt_df = counts_table(all_cls, SPECIFIC_GENES, cols['cell_line'], symbol_to_row, expr_mat, model_name_to_index)
    input_df = input_df.merge(cnt_df, on=cols['cell_line'], how='left')

    has_seq = input_df[cols['gene_sequence']].astype(str).str.len().ge(10)
    has_count = input_df['Antigen_Receptor_mRNA_Read_Counts'].notna()
    combined = input_df[has_seq & has_count].reset_index(drop=True)
    if combined.empty:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output_csv, index=False)
        logging.warning('No valid rows after sequence/count filtering')
        return combined

    for g in SPECIFIC_GENES:
        raw = f'{g}_mRNA_count'
        scaled = f'{g}_mRNA_count_scaled'
        combined[scaled] = np.nan
        ok = combined[raw].notna()
        if ok.any():
            combined.loc[ok, scaled] = scaler_rc.transform(combined.loc[ok, raw].values.reshape(-1, 1).astype(np.float32)).ravel()

    pca_size = pca_tx.n_components_
    pca_cols = [f'NN_mRNA_Cell_Line_PCA_{i}' for i in range(pca_size)]
    pca_rows = []
    for cl in tqdm(combined[cols['cell_line']].astype(str).unique(), desc='Cell-line PCA', unit='cell'):
        pc = cell_line_pca(cl, model_name_to_index, expr_mat, scaler_tx, pca_tx)
        if pc is not None:
            pca_rows.append([cl] + pc.tolist())
    pca_df = pd.DataFrame(pca_rows, columns=[cols['cell_line']] + pca_cols)
    combined = combined.merge(pca_df, on=cols['cell_line'], how='inner')
    if combined.empty:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output_csv, index=False)
        logging.warning('No valid rows after transcriptomic PCA')
        return combined

    logging.info('Loading protein-intensity model')
    pi_model = ProteinIntensityModel.load_from_checkpoint(
        str(_path(cfg, 'protein_intensity_checkpoint')),
        esm_embedding_dim=pca_prot.n_components_,
        mrna_embedding_dim=pca_size,
        read_count_dim=1,
        strict=False,
    ).to(device)
    pi_model.eval()

    cl_pca_df = combined[[cols['cell_line']] + pca_cols].drop_duplicates(cols['cell_line'])
    cl_list = cl_pca_df[cols['cell_line']].astype(str).tolist()
    cl_to_row = {cl: i for i, cl in enumerate(cl_list)}
    pca_mat = cl_pca_df[pca_cols].to_numpy(dtype=np.float32)

    pair_df = combined[[cols['gene'], cols['cell_line']]].drop_duplicates().reset_index(drop=True)
    pair_df[cols['gene']] = pair_df[cols['gene']].astype(str)
    pair_df[cols['cell_line']] = pair_df[cols['cell_line']].astype(str)

    gene_to_emb = {}
    for g in tqdm(pair_df[cols['gene']].unique(), desc='Main gene ESM PCA', unit='gene'):
        seq = combined.loc[combined[cols['gene']].astype(str) == str(g), 'Main_Gene_Sequence'].iloc[0]
        emb = esm_pca_for_sequence(seq, esm_model, batch_converter, alphabet, device, scaler_prot, pca_prot, repr_layer)
        if emb is not None:
            gene_to_emb[str(g)] = emb

    pair_df = pair_df[pair_df[cols['gene']].map(lambda g: str(g) in gene_to_emb)].reset_index(drop=True)
    sym_idx = pair_df[cols['gene']].map(symbol_to_row).astype(int).to_numpy()
    cl_idx = pair_df[cols['cell_line']].map(model_name_to_index).astype(int).to_numpy()
    rc_raw = expr_mat[sym_idx, cl_idx]
    rc_scaled = scaler_rc.transform(rc_raw.reshape(-1, 1)).astype(np.float32).ravel()
    E = np.stack([gene_to_emb[str(g)] for g in pair_df[cols['gene']]], axis=0)
    P = np.stack([pca_mat[cl_to_row[str(cl)]] for cl in pair_df[cols['cell_line']]], axis=0)

    preds_pi, x_mrna, x_rc = _predict_pi_batches(pi_model, E, P, rc_scaled, device, pi_batch_size)
    pair_df['Predicted_Protein_Intensity'] = preds_pi

    mrna_cols = [f'NN_mRNA_Cell_Line_Embedding_{i}' for i in range(x_mrna.shape[1])]
    mrna_by_cl = pd.DataFrame(x_mrna, columns=mrna_cols).assign(**{cols['cell_line']: pair_df[cols['cell_line']]}).groupby(cols['cell_line'], as_index=False).first()
    rc_cols = [f'NN_Gene_mRNA_Counts_Embedding_{i}' for i in range(x_rc.shape[1])]
    pair_embed = pd.concat([
        pair_df[[cols['gene'], cols['cell_line'], 'Predicted_Protein_Intensity']].reset_index(drop=True),
        pd.DataFrame(x_rc, columns=rc_cols),
    ], axis=1)
    combined = combined.merge(pair_embed, on=[cols['gene'], cols['cell_line']], how='left')
    combined = combined.merge(mrna_by_cl, on=cols['cell_line'], how='left')

    for g in tqdm(SPECIFIC_GENES, desc='Resistance gene PI', unit='gene'):
        ridx = symbol_to_row.get(g)
        e2 = gene2esm.get(g)
        if ridx is None or e2 is None or np.isnan(e2).any():
            combined[f'Predicted_Protein_Intensity_{g}'] = np.nan
            continue
        idxs = [model_name_to_index[cl] for cl in cl_list]
        raw = expr_mat[ridx, idxs]
        scaled = scaler_rc.transform(raw.reshape(-1, 1)).astype(np.float32).ravel()
        vals = np.empty((len(cl_list),), dtype=np.float32)
        for start in range(0, len(cl_list), pi_batch_size):
            end = min(start + pi_batch_size, len(cl_list))
            with torch.no_grad():
                out, _, _, _ = pi_model(
                    torch.tensor(np.repeat(e2[None, :], end - start, axis=0), dtype=torch.float32, device=device),
                    torch.tensor(pca_mat[start:end], dtype=torch.float32, device=device),
                    torch.tensor(scaled[start:end], dtype=torch.float32, device=device).unsqueeze(1),
                )
            vals[start:end] = out.squeeze(1).cpu().numpy().astype(np.float32)
        combined[f'Predicted_Protein_Intensity_{g}'] = combined[cols['cell_line']].map(dict(zip(cl_list, vals))).astype(np.float32)

    gcols = [f'Gene_PCA_esm_embeddings_{i}' for i in range(pca_prot.n_components_)]
    rows = []
    for g, emb in gene_to_emb.items():
        rows.append([g] + emb.tolist())
    gene_emb_df = pd.DataFrame(rows, columns=[cols['gene']] + gcols)
    combined = combined.merge(gene_emb_df, on=cols['gene'], how='left')

    numeric_like = list(RDKit_DESCRIPTOR_COLUMNS)
    numeric_like += [c for c in combined.columns if c.startswith('MACCS_')]
    numeric_like += [c for c in combined.columns if c.startswith('NN_mRNA_Cell_Line_Embedding_')]
    numeric_like += [c for c in combined.columns if c.startswith('NN_Gene_mRNA_Counts_Embedding_')]
    numeric_like += [c for c in combined.columns if c.startswith('Gene_PCA_esm_embeddings_')]
    numeric_like += [c for c in combined.columns if c.endswith('_mRNA_count_scaled')]
    numeric_like += [c for c in combined.columns if c.startswith('Predicted_Protein_Intensity')]
    numeric_like += ['payload_mw', 'adc_mw', 'Antigen_Receptor_mRNA_Read_Counts']
    numeric_like = [c for c in dict.fromkeys(numeric_like) if c in combined.columns]
    combined[numeric_like] = combined[numeric_like].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    combined.fillna(0, inplace=True)

    params = cfg['cnn_params']
    cnn = ADCModel.load_from_checkpoint(
        str(_path(cfg, 'cnn_checkpoint')),
        feature_params=params,
        classifier_params=params,
        lr=params['lr'],
        boundary=float(cfg.get('threshold_nm', 10.0)),
        optimizer_type=params.get('optimizer_type', 'adam'),
        loss=cfg.get('training', {}).get('loss', 'boundary_weighted_bce'),
        strict=False,
    ).to(device)
    cnn.eval()

    ds = ADCPredictionDataset(combined)
    batch_size = int(pred_cfg.get('cnn_batch_size', max(128, params.get('batch_size', 32))))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    for batch in tqdm(dl, total=math.ceil(len(combined) / batch_size), desc='CNN inference', unit='batch'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = cnn(batch)
        preds.append(out['probs'].detach().cpu())
    combined['predicted_probability'] = torch.cat(preds, dim=0).numpy()

    for cutoff in pred_cfg.get('probability_cutoffs', [0.5, 0.6]):
        suffix = str(cutoff).replace('.', '_')
        combined[f'predicted_label_{suffix}'] = (combined['predicted_probability'] > float(cutoff)).astype(int)

    combined['IC50_prediction'] = np.where(
        combined['predicted_label_0_5'] == 1,
        'Pred < 10 nM',
        'Pred ≥ 10 nM',
    )

    def target(row):
        if row.get(cols['is_tubulin'], 0) == 1:
            return 'tubulin'
        if row.get(cols['is_dna'], 0) == 1:
            return 'dna'
        if row.get(cols['is_not_tubulin_dna'], 0) == 1:
            return 'topoisomerase'
        return 'unknown'
    combined['intracellular_target'] = combined.apply(target, axis=1)

    if 'InVitro_Equalized_IC50_Value' in combined.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        y = (combined['InVitro_Equalized_IC50_Value'].astype(float) < 10.0).astype(int)
        p = combined['predicted_probability']
        pred = combined['predicted_label_0_5']
        logging.info('Accuracy %.4f', accuracy_score(y, pred))
        logging.info('Precision %.4f', precision_score(y, pred, zero_division=0))
        logging.info('Recall %.4f', recall_score(y, pred, zero_division=0))
        logging.info('F1 %.4f', f1_score(y, pred, zero_division=0))
        if len(np.unique(y)) > 1:
            logging.info('AUC %.4f', roc_auc_score(y, p))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logging.info('Wrote %s', out_path)
    return combined
