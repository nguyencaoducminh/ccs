# Transformer for predicting peptide collitional cross-section (CCS)
---
This repository contains two transformers for predicting peptide CCS create based on transformer for predicting retention time (https://github.com/aiproteomics/rt) by modifying the Pytorch version of the tranformer (https://github.com/chauttm/rt)

## Data

- Ionmob datasets: data/ionmob

- Meier 2021 dataset: data/meier_2021

## Transformers

- CCS Transformer using only amino acids sequence: model.py

- CCS Transformer using amino acids sequence and charge: modelcharge.py

## Usage

### Training
```
# Transformer using only peptide's sequence
python train.py --compile=False --batch_size=64 --n_layer=4 --n_head=8 --n_embd=64 --dropout=0.1 --dff=256 --epochs=10 --dataset=meier

# Transformer using peptide's sequence and charge
python traincharge.py --compile=False --batch_size=64 --n_layer=4 --n_head=8 --n_embd=64 --dropout=0.1 --dff=256 --epochs=10 --dataset=meier
```

### Predicting
```
# Transformer using only peptide's sequence
python predict.py --dataset=meier --model_dir=out\meier-b64-dm64-df256-nl4-nh8-dr0.1-ep10

# Transformer using peptide's sequence and charge
python predictcharge.py --dataset=meier --model_dir=out\meier-charge-b64-dm64-df256-nl4-nh8-dr0.1-ep10
```