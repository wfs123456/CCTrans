# CCTrans: Simplifying and Improving Crowd Counting with Transformer(Code reproduction)
* Code reproduction
* Original paper [Link](https://arxiv.org/pdf/2109.14483.pdf)

## Overview
* Presentate only the experiment on dataset ShanghaiTech Part A (loss: DM-Count)
* ShanghaiTech Part A 

| Code      | MAE   | MSE      |
|-----------|-------|-------|
| PAPER     | 54.8  | 86.6  |
| This code | 54.20 | 88.97 |

Our code reaches this result with the standard hyperparameter set in code. Trained with batch-size=8 for around 1500 epoch(as said in the paper). Best validation at around epoch 606
# code framework
* adopt code of DM-Count.
* [link](https://github.com/cvlab-stonybrook/DM-Count)

# Pre-trained weights
* Download  pretrained weights for ```alt_gvt_large.pth```  [link](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view). Move the file under ```model_weights``` folder

# Training
Take a look at the arguments accepted by ```train.py```
* update root "data-dir" in ./train.py.
* [new] Added [wandb](https://wandb.ai/) integration. If you want to log with wandb, set ```--wandb 1``` in ```train.py``` after having logged in to wandb (```wandb login``` in console)
* launch with ```python train.py```

# Testing
* python test_image_patch.py
* Due to crop training with size of 256x256, the validation image is divided into several patches with size of 256x256, and the overlapping area is averaged.
* Download the pretrained model from Baidu-Disk(提取码: se59) [link](https://pan.baidu.com/s/16qY_cFIUAUaDRsdr5vNsWQ)

# Visualization
* python vis_densityMap.py
* save to ./vis/part_A_final

# TensorRT
Require [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

See ```torch_to_trt.py```

# Environment
	See requirements.txt
	


