# CCTrans: Simplifying and Improving Crowd Counting with Transformer(Code reproduction)
* Code reproduction
* Original paper [Link](https://arxiv.org/pdf/2109.14483.pdf)

## Overview
* Presentate only the experiment on dataset ShanghaiTech Part A (loss: DM-Count)
* ShanghaiTech Part A result:  MAE/MSE 54.8/86.6(original paper)        MAE/MSE 54.3/91.6(reproduction)

# code framework
* adopt code of DM-Count.
* [link](https://github.com/cvlab-stonybrook/DM-Count)

# Training
Take a look at the arguments accepted by ```train.py```
* update root "data-dir" in ./train.py.
* load pretrained weights of ImageNet-1k in ./Networks/ALTGVT.py.
* pretrained weights [link](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view)
* [new] Added [wandb](https://wandb.ai/) integration. If you want to log with wandb, set ```--wandb 1``` in ```train.py``` after having logged in to wandb (```wandb login``` in console)
* launch with ```python train.py```

# Testing
* python test_image_patch.py
* Due to crop training with size of 256x256, the validation image is divided into several patches with size of 256x256, and the overlapping area is averaged.
* Download the pretrained model from Baidu-Disk(提取码: se59) [link](https://pan.baidu.com/s/16qY_cFIUAUaDRsdr5vNsWQ)

# Visualization
* python vis_densityMap.py
* save to ./vis/part_A_final

# Environment
	See requirements.txt
	


