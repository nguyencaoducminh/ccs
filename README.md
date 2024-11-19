# Experiment with a transformer architecture for retention time prediction


![Transformer for retention time prediction](images/graphical-abstract.png)


This repository contains our experiments with the transformer architecture for retention time prediction in liquid chromatography mass spectrometry-based proteomics.

## Data

- Prosit: data/prosit_rt_updated/*

- AutoRT: data/PXD006109*

- DeepDIA: data/PXD005573*

- Independent test set: data/PXD019038*

- Phospho dataset for training, validation, and hold out: data/HumanPhosphoproteomeDB_rt*

- Phospho datasets for testing: data/test-PXD006637.csv, data/test-PXD013453.csv, and data/test-PXD019113.csv

## Trained models

- Prosit: models/prosit/

- AutoRT: models/autort/

- DeepDIA: models/deepdia/

- phosho: models/phospho/

## Example usages

We used Python 3.10 and Pytorch.

### Tunning  (TO BE MODIFIED)
```
python rt.py tune -data autort -logs logs-random-search-autort -epochs 50 -n_layers 6 8 10 12 -n_heads 2 4 8 -dropout 0.05 0.1 0.15 -batch_size 64 128 256 512 1024 -d_model 256 512 768 -d_ff 256 512 768 1024 -seed 0 -n_random_samples 100
```
This command will perform a random search for hyperparameter settings. The result is a text output containing epoch, loss, and validation loss for further analysis in R. There is also a log folder for TensorBoard visualization.

### Training

```
python train.py --device=cpu --compile=False --batch_size=64 --n_layer=4 --n_head=8 --n_embd=64 --lr_decay_iters=200 --dropout=0.1 --dff=256 --epochs=10 --dataset=autort
```
This command will produce a trained model with the specified hyperparameters.

### Testing on holdout data

```
python .\predict.py --device=cpu --dataset=autort --model_dir=out\deepdia-b64-dm64-df256-nl4-nh8-dr0.1-ep10
```
The predicted values are in column 'y_pred' in the output tab-separated-value file. The measured values are in column 'y'. One has to use the holdout data corresponding to the trained model.

### Predicting retention time for a new set of peptides (TO BE MODIFIED)

```
python rt.py predict [model_dir/] -input peptides.csv -output peptides-rt.txt
```
The input peptides.csv is a comma-separated-value text file containing two columns named 'sequence' and 'rt'. When measured rt values are not available, one can fill the rt column with 0. The predicted values are in column 'y_pred' in the output tab-separated-value peptides-rt.txt file. The input rt values are copied to column 'y' in the output.
