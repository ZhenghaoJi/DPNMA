# DPNMA-crowd-counting

This is official implement for Dual Path Networks With Multi-scale Non-local Attention for Crowd Counting

## Prerequisite

Python 3.7

Pytorch 1.4.0

## Code structure

`density_map.py` To generate the density map and attention map. 
root = r'/data/CrowdCount/'
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

`dataset.py` and `transforms.py` For data preprocess and augmentation. 

`models_nonlocal.py` The structure of the network. 

`train_nonlocal.py` To train the model. 

`eval_nonlocal.py` To test the model. 

## Train & Test

For training, run
in dataset.py line 51 modify to return image, gt
sh train_nonlocal.sh

For testing, run
in dataset.py line 50 modify to return image, gt,density,self.image_list[index]

sh eval_nolocal.sh

## Result
MAE 58.3
MSE 92.6
