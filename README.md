# AMM ADC 10 nM Prediction Pipeline

This repository contains the 10 nM AMM ADC prediction workflow used for ADC prediction. It takes ADC payload/linker SMILES, antigen information, and cell-line transcriptomic context, then produces a probability and binary call for predicted activity below 10 nM.

The default configuration is in `config/default.yaml`; model checkpoints are available on Zenodo [link].

## Install

A clean conda environment is recommended.

```bash
conda create -n amm_adc_10nm python=3.9 -y
conda activate amm_adc_10nm
pip install -r requirements.txt
```

On systems where `rdkit-pypi` is unreliable, install RDKit from conda-forge first:

```bash
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

The ESM2 model used by default is `esm2_t33_650M_UR50D`. The first run may download the model weights to the local Torch cache unless they are already present.


## Input format

CSV prediction input should include at least:

```text
SMILES,CellLine,GeneSymbol,ProteinSequence,Drug_Antibody_Ratio_(DAR),Is_Tubulin_Target,Is_DNA_Target,Is_not_Tubulin_DNA_Target
```

`ProteinSequence` can be left blank if a valid `UniProt_ID` is supplied, or if the UniProt lookup for the gene symbol is acceptable. For reproducible runs, it is better to provide the sequence directly.

## Run prediction

```bash
python scripts/predict.py \
  --config config/default.yaml \
  --input-csv data/examples/example_input.csv \
  --output-csv outputs/predictions_10nM.csv
```

The output includes:

```text
predicted_probability
predicted_label_0_5
predicted_label_0_6
IC50_prediction
Predicted_Protein_Intensity
```

`IC50_prediction` is reported only for the 10 nM decision boundary:

```text
Pred < 10 nM
Pred ≥ 10 nM
```

## Manual mode

```bash
python scripts/predict.py \
  --config config/default.yaml \
  --smiles "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@H](C(C)C)N(C)C(=O)OCc4ccc(NC(=O)[C@H](CCCNC(N)=O)NC(=O)[C@@H](NC(=O)CCCCCN3C(=O)CC(S)C3=O)C(C)C)cc4)C(C)C" \
  --cell-line MCF-7 \
  --gene-symbol CD33 \
  --dar 3.0 \
  --is-tubulin 0 \
  --is-dna 0 \
  --is-not-tubulin-dna 1 \
  --output-csv outputs/manual_10nM.csv
```

To score all cell lines present in the transcriptomic HDF5:

```bash
python scripts/predict.py \
  --config config/default.yaml \
  --smiles "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@H](C(C)C)N(C)C(=O)OCc4ccc(NC(=O)[C@H](CCCNC(N)=O)NC(=O)[C@@H](NC(=O)CCCCCN3C(=O)CC(S)C3=O)C(C)C)cc4)C(C)C" \
  --gene-symbol CD33 \
  --all-cell-lines \
  --output-csv outputs/all_cell_lines_10nM.csv
```

## Re-training the 10 nM CNN

Prepared feature tables are expected under `data/processed/` by default. To train with the default 10 nM model parameters:

```bash
python scripts/train_cnn.py --config config/default.yaml
```

Outputs are written to the configured checkpoint and log directories.


## Citation



## License


