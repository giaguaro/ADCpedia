from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, Descriptors
from tqdm.auto import tqdm
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

from .constants import RDKit_DESCRIPTOR_COLUMNS

_desc_generator = MakeGenerator(("rdkit2dhistogramnormalized",))


@lru_cache(maxsize=8192)
def _desc_cached(smiles):
    d = _desc_generator.process(smiles)
    if d and d[0]:
        return tuple(d[1:])
    return None


def rdkit_descriptors(smiles):
    vals = _desc_cached(str(smiles))
    if vals is None:
        return [np.nan] * len(RDKit_DESCRIPTOR_COLUMNS)
    return list(vals)


@lru_cache(maxsize=8192)
def _maccs_cached(smiles, size):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((int(size),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return tuple(arr.tolist())


def maccs_fingerprint(smiles, size=167):
    vals = _maccs_cached(str(smiles), int(size))
    if vals is None:
        return [np.nan] * int(size)
    return list(vals)


def payload_mw(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.nan
    return Descriptors.MolWt(mol)


def build_chem_table(unique_smiles, smiles_col, maccs_size=167):
    rows = []
    for smi in tqdm(list(unique_smiles), desc='Chemistry', unit='smi'):
        row = {smiles_col: smi}
        vals = rdkit_descriptors(smi)
        for c, v in zip(RDKit_DESCRIPTOR_COLUMNS, vals):
            row[c] = v
        bits = maccs_fingerprint(smi, maccs_size)
        for i, bit in enumerate(bits):
            row[f'MACCS_{i}'] = bit
        row['payload_mw'] = payload_mw(smi)
        rows.append(row)
    return pd.DataFrame(rows)
