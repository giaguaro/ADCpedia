import logging

import numpy as np
import pandas as pd


def load_transcriptomic_data(hdf5_file):
    logging.info('Reading transcriptomic data from %s', hdf5_file)
    df = pd.read_hdf(hdf5_file, key='rnaseq_data')
    idx_map = pd.read_hdf(hdf5_file, key='model_name_to_index')

    model_name_to_index = {}
    if isinstance(idx_map, pd.DataFrame):
        cols = idx_map.columns
        if 'model_name' in cols and 'index' in cols:
            for _, r in idx_map.iterrows():
                model_name_to_index[str(r['model_name']).strip()] = int(r['index'])
        else:
            c0, c1 = cols[0], cols[1]
            for _, r in idx_map.iterrows():
                model_name_to_index[str(r[c0]).strip()] = int(r[c1])
    elif isinstance(idx_map, pd.Series):
        for k, v in idx_map.items():
            model_name_to_index[str(k).strip()] = int(v)
    else:
        raise RuntimeError('model_name_to_index has an unsupported shape')

    return df, model_name_to_index


def prepare_rnaseq(df):
    symbols = df['symbol'].astype(str).values
    symbol_to_row = {s: i for i, s in enumerate(symbols)}
    values = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    expr_mat = values.to_numpy(dtype=np.float32, copy=False)
    return symbol_to_row, expr_mat


def fetch_count(symbol, cell_line, symbol_to_row, expr_mat, model_name_to_index):
    ridx = symbol_to_row.get(str(symbol))
    cidx = model_name_to_index.get(str(cell_line).strip())
    if ridx is None or cidx is None:
        return np.nan
    val = expr_mat[ridx, cidx]
    return float(val) if np.isfinite(val) else np.nan


def cell_line_pca(cell_line, model_name_to_index, expr_mat, scaler, pca):
    idx = model_name_to_index.get(str(cell_line).strip())
    if idx is None:
        return None
    vals = expr_mat[:, idx]
    scaled = scaler.transform(vals.reshape(1, -1))
    return pca.transform(scaled)[0]


def counts_table(cell_lines, genes, cell_col, symbol_to_row, expr_mat, model_name_to_index):
    rows = []
    for cl in cell_lines:
        d = {cell_col: cl}
        cidx = model_name_to_index.get(str(cl).strip())
        for g in genes:
            ridx = symbol_to_row.get(g)
            if ridx is None or cidx is None:
                d[f'{g}_mRNA_count'] = np.nan
            else:
                d[f'{g}_mRNA_count'] = float(expr_mat[ridx, cidx])
        rows.append(d)
    return pd.DataFrame(rows)


def name_resolver(model_name_to_index):
    d = {}
    for key in model_name_to_index:
        norm = ''.join(ch for ch in str(key).upper() if ch.isalnum())
        d.setdefault(norm, key)
    return d


def resolve_name(name, resolver):
    norm = ''.join(ch for ch in str(name).upper() if ch.isalnum())
    return resolver.get(norm)
