# AMM Integrated ADC Prediction Pipeline

This repository contains a single main script **`AMM_model_predict.py`** and its supporting model/data files. The script performs an end-to-end workflow that:

1. Reads SMILES structures and (optionally) gene information from a CSV (or uses manually provided arguments).
2. Generates chemical descriptors using RDKit and Descriptastorus-based 2D descriptors.
3. Computes additional feature embeddings such as MACCS fingerprints, transcriptomic embeddings, and learned ESM protein embeddings (via [Facebook’s ESM2 model](https://github.com/facebookresearch/esm)).
4. Predicts protein intensity with a trained PyTorch Lightning model (`ProteinIntensityModel`).
5. Uses a final convolutional neural network model to classify whether an antibody–drug conjugate (ADC) is predicted to be effective below certain IC50 thresholds.

With this pipeline, you can take a list of candidate ADC constructs, supply the SMILES plus either the gene symbols or protein sequences, combine with cell-line transcriptomic data, and get a final binary prediction of ADC activity (together with intermediate features such as predicted protein intensities).

---

## Table of Contents

1. [Overview](#overview)  
2. [Installation and Dependencies](#installation-and-dependencies)  
3. [Usage](#usage)  
    - [CSV Input Mode](#csv-input-mode)  
    - [Manual Input Mode](#manual-input-mode)  
    - [Important Arguments](#important-arguments)  
    - [Example Commands](#example-commands)  
4. [Citation](#citation)  
5. [License](#license)

---

## Overview

**`AMM_model_predict.py`** orchestrates a multi-step prediction pipeline:

- **Descriptor Generation**: SMILES are converted to RDKit-based descriptors, MACCS fingerprints, etc.
- **Feature Assembly**: The script integrates:  
  - **Chemical features**: 2D descriptors, molecular weight calculations, etc.  
  - **Gene/Protein features**: ESM2 embeddings (for the target protein), scaled/PCA-transformed transcriptomic profiles for the cell line, scaled read counts, etc.
- **Protein Intensity Prediction**: A trained regression model (`ProteinIntensityModel`) estimates protein abundance from combined ESM embeddings, cell-line transcriptomic embeddings, and read counts.
- **Final CNN Classification**: A second PyTorch Lightning model performs a binary classification of ADC efficacy using the assembled feature vectors.

Use either:

- A CSV file with columns specifying SMILES, gene symbol, cell line, etc., **OR**
- Command-line arguments for a single SMILES and multiple `(gene_symbol, cell_line)` pairs.

---

## Installation and Dependencies

### 1. Clone or Download This Repository

```bash
git clone https://github.com/username/amm_adc_pipeline.git
cd amm_adc_pipeline
```
```

### 2. Install Required Packages

Create a new Python environment (recommended) and install dependencies:

```bash
conda create -n amm_adc python=3.9 -y
conda activate amm_adc

# Required packages
pip install numpy pandas scipy scikit-learn joblib tqdm
pip install rdkit-pypi
pip install descriptastorus
pip install torch torchvision torchaudio
pip install pytorch-lightning

# ESM (for protein embeddings). We use 650M model. 
pip install fair-esm

```

---

## Usage

Run the script via:

```bash
python AMM_model_predict.py --help
```

to see all available arguments.

### CSV Input Mode

In CSV mode, you provide an input CSV file with these (minimum) columns:

- `SMILES`  
- `CellLine`  
- `GeneSymbol`  
- `ProteinSequence` (optional if you have a corresponding `UniProt_ID` or rely on automatic UniProt queries)  
- `Drug_Antibody_Ratio_(DAR)`  
- `Is_Tubulin_Target`  
- `Is_DNA_Target`  
- `Is_not_Tubulin_DNA_Target`

You then specify:

```bash
python AMM_model_predict.py \
  --input_csv my_input.csv \
  --output_csv predictions.csv \
  --cnn_model_checkpoint <cnn_model_checkpoint>.ckpt \
  --protein_intensity_model_checkpoint epoch=24-val_rmse=0.69-val_mse=0.48-val_r2=0.87.ckpt
```

The script will:

1. Read each row in `my_input.csv`.  
2. Generate descriptors/fingerprints.  
3. Fetch transcriptomic data.  
4. (If needed) Attempt to look up sequences from UniProt if `ProteinSequence` is empty.  
5. Predict protein intensities, then run the final CNN model to output:
   - `predicted_probability` (the model’s probability of ADC potency).
   - Binary labels at different cutoffs, e.g. `predicted_label_0_5`.

### Manual Input Mode

Instead of a CSV, you can supply a single SMILES (payload), plus multiple `(cell_line, gene_symbol)` pairs, for example:

```bash
python AMM_model_predict.py \
  --smiles "C[C@@H](C(N[C@H](C)[C@H](C1=CC=CC=C1)O)=O)[C@H]([C@@](CCC2)([H])N2C(C[C@@H](OC)[C@H]([C@@H](C)CC)N(C)C([C@H](C(C)C)NC([C@H](C(C)C)N(C)C(OCC(C=C3)=CC=C3NC([C@H](C)NC([C@H](C(C)C)NC(CCCCCN(C4=O)C(C=C4)=O)=O)=O)=O)=O)=O)=O)=O)OC" \
  --cell_line MDA-MB-468 CALU-6 \
  --gene_symbol ERBB2 \
  --dar 3.4 \
  --is_tubulin 0 \
  --is_dna 1 \
  --is_not_tubulin_dna 0 \
  --uniprot_id P08183 P45916 \
  --manual_gene_sequence MTEITAAMVK... \
  --output_csv manual_output.csv \
  --cnn_model_checkpoint adcpedia_checkpoints/<threshold>nM/<cnn_model_checkpoint>.ckpt \
  --protein_intensity_model_checkpoint epoch=24-val_rmse=0.69-val_mse=0.48-val_r2=0.87.ckpt
```

This example:

- Takes one SMILES string.
- Applies the same SMILES to two cell lines (`MDA-MB-468` and `CALU-6`).
- Uses two gene symbols (`ERBB2`) in combination with the cell lines.
- Optionally supplies matching UniProt IDs or manual sequences for each gene.

The script will generate separate rows internally and produce predictions in a single output CSV.

### Important Arguments

- **`--cnn_model_checkpoint`**  
  Required: path to the trained CNN `.ckpt` for final ADC classification.

- **`--num_workers`**  
  Number of parallel worker processes for descriptor generation.

### Example Commands

1. **CSV mode**:
   ```bash
   python AMM_model_predict.py \
     --input_csv example_input.csv \
     --output_csv example_output.csv \
     --cnn_model_checkpoint myCNN_10nM.ckpt \
     --protein_intensity_model_checkpoint epoch=24-val_rmse=0.69-val_mse=0.48-val_r2=0.87.ckpt
   ```

2. **Manual mode** (single SMILES, multiple `(cell_line, gene_symbol)` combos):
   ```bash
   python AMM_model_predict.py \
     --smiles "C[C@@H](C(N[C@H](C)[C@H](C1=CC=CC=C1)O)=O)[C@H]([C@@](CCC2)([H])N2C(C[C@@H](OC)[C@H]([C@@H](C)CC)N(C)C([C@H](C(C)C)NC([C@H](C(C)C)N(C)C(OCC(C=C3)=CC=C3NC([C@H](C)NC([C@H](C(C)C)NC(CCCCCN(C4=O)C(C=C4)=O)=O)=O)=O)=O)=O)=O)=O)OC" \
     --cell_line MCF-7 \
     --gene_symbol CD33 \
     --dar 3.0 \
     --is_tubulin 0 \
     --is_dna 0 \
     --is_not_tubulin_dna 1 \
     --output_csv manual_test.csv \
     --cnn_model_checkpoint myCNN_5nM.ckpt
   ```

Check your resulting CSV file for columns like `predicted_probability` and `predicted_label_0_5`.

---

## Citation

If you use this code or the models in your research, please cite:


## License

