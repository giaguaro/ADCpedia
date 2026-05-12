import logging
import time

import requests


def fetch_sequence_by_uniprot_id(uniprot_id, retries=3, delay=2):
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta'
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                lines = r.text.strip().splitlines()
                return ''.join(lines[1:])
            logging.warning('UniProt %s returned %s', uniprot_id, r.status_code)
        except Exception as e:
            logging.warning('UniProt error for %s: %s', uniprot_id, e)
        time.sleep(delay)
    return None


def fetch_canonical_sequence(gene_symbol, retries=3, delay=2):
    url = (
        'https://rest.uniprot.org/uniprotkb/search?'
        f'query=gene_exact:{gene_symbol}+AND+reviewed:true'
        '&format=tsv&fields=accession,sequence'
    )
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                lines = r.text.strip().splitlines()
                for line in lines[1:]:
                    parts = line.split('\t')
                    if len(parts) >= 2 and len(parts[1]) > 10:
                        return parts[1]
            else:
                logging.warning('UniProt search %s returned %s', gene_symbol, r.status_code)
        except Exception as e:
            logging.warning('UniProt error for %s: %s', gene_symbol, e)
        time.sleep(delay)
    return None


def fill_missing_sequences(df, gene_col, seq_col, fetch_missing=True):
    cache = {}
    if seq_col not in df.columns:
        df[seq_col] = ''
    for i, row in df.iterrows():
        seq = row.get(seq_col, '')
        if isinstance(seq, str) and len(seq) >= 10:
            continue
        if not fetch_missing:
            continue
        gene = str(row.get(gene_col, '')).strip()
        if not gene:
            continue
        uniprot_id = row.get('UniProt_ID')
        key = str(uniprot_id).strip() if isinstance(uniprot_id, str) and uniprot_id.strip() else gene
        if key not in cache:
            found = None
            if isinstance(uniprot_id, str) and uniprot_id.strip():
                found = fetch_sequence_by_uniprot_id(uniprot_id.strip())
            if not found:
                found = fetch_canonical_sequence(gene)
            cache[key] = found
        if cache[key] and len(cache[key]) >= 10:
            df.at[i, seq_col] = cache[key]
    return df
