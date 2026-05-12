# Input schema

A minimal prediction input looks like:

```csv
Name,SMILES,CellLine,GeneSymbol,ProteinSequence,Drug_Antibody_Ratio_(DAR),Is_Tubulin_Target,Is_DNA_Target,Is_not_Tubulin_DNA_Target,UniProt_ID
example_adc,CCO,MCF-7,CD33,,3.0,0,0,1,P20138
```

For reproducible runs, provide the protein sequence directly rather than relying on UniProt at runtime.
