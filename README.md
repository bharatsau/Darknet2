# Darknet2

## Introduction
This repository is a modified version of original darknet framework(https://github.com/pjreddie/darknet).

Famous object detection works- YoloV2 and YoloV3 was developed in darknet framework.

While using darknet framework for training new networks for detection using knowledge distillation method, I encountered several issues. I had to modify and add new features as a result.

## New Features:

1. **Transfer Learning:** It supports transfer learning for selected layers. It has been developed in caffe-style, i.e., if layer name is same in both the cfg, then only transfer of weights/paramters will happen.

2. **New Layers:** CPU and GPU implementation of channelwise-L2-normalization, channelwise-sum, eltwise sum(just like caffe eltwise).

3. **Activation Functions:** negative_ReLU, absolute, square activation functions for generating attention maps(https://arxiv.org/pdf/1612.03928.pdf). 

4. **Other Functionalities:** Knowledge distillation(kd_regressor.c), multi_loss detection training(multiloss_det.c).



## Sample Execution Commands: 
Need to add some sample running commands. Will add soon.


## Debugging Guides: 
Will add soon. Please contact me at bharatsau1987@gmail.com if you need any help.




