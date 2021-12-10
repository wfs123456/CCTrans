# CCTrans: Simplifying and Improving Crowd Counting with Transformer(Code reproduction)
* Code reproduction
* Original paper [Link](https://arxiv.org/pdf/2103.05242.pdf)

## Overview
* Presentate only the experiment on dataset ShanghaiTech Part A (loss: DM-Count)
* ShanghaiTech Part A result:  MAE/MSE 54.8/86.6(original paper)        MAE/MSE 54.3/91.6(reproduction)

# code framework
* adopt code of DM-Count.
* [link](https://github.com/cvlab-stonybrook/DM-Count)

# Training
* update root "data-dir" in ./train.py.
* load pretrained weights of ImageNet-1k in ./Networks/ALTGVT.py.
* pretrained weights [link](https://github.com/Meituan-AutoML/Twins/alt_gvt_large.pth)
* python train.py

# Testing
* python test_image_patch.py
* Due to crop training with size of 256*256,  the validation image is divided into several patches with size of 256*256, and the overlapping area is averaged.

# Visualization
* python vis_densityMap.py
* save to ./vis/part_A_final

# Environment
	python >=3.6 
	torch==1.7.0
	torchvision==0.8.1
	timm==0.3.2
	opencv-python >=4.0
	scipy >=1.4.0
	pillow >=7.0.0
	imageio >=1.18
	


