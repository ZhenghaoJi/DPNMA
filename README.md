# DPNMA-crowd-counting

This is official implement for Dual Path Networks With Multi-scale Non-local Attention for Crowd Counting

## Prerequisite

Python 3.7

Pytorch 1.4.0

## Code structure

`density_map.py` To generate the density map and attention map. 

`dataset.py` and `transforms.py` For data preprocess and augmentation. 

`models_nonlocal.py` The structure of the network. 

`train_nonlocal.py` To train the model. 

`eval_nonlocal.py` To test the model. 

## Train & Test

For training, run

sh train_nonlocal.sh

For testing, run

sh eval_nolocal.sh

## Result
MAE 58.3
MSE 92.6
