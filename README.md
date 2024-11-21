# Denoising-Aware Contrastive Learning for Noisy Time Series (DECL)

## 1. Introduction
This repository contains code for the paper "[Denoising-Aware Contrastive Learning for Noisy Time Series](https://www.ijcai.org/proceedings/2024/0624.pdf)" (IJCAI 2024).


## 2. Usage
### Requirements:
+ torch==1.7.1
+ python==3.7.16

See requirements.txt for details.


### Datasets:
The code assumes that the data has been augmented and saved into the data folder.

For example, the folder of ./PTB/ contains all the raw data, the folder of ./PTB_denoise1/ contains all the denoised data using the denoiser m1, the folder of ./PTB_noisy1/ contains all the noise-enhanced data corresponding to denoiser m1, and the folder of ./PTB_Gaussian/ contains all the data induced with Gaussian noise.

The meta information of each time series sample is in the label.csv file.


### Example:
Please modify the 'data_path' in the code to adapt to the path of your data folder.

If using a few labels for training, please split data twice: one for pre-training and one for linear evaluation.

+ python run_preprocess.py --trn_ratio 0.4 --val_ratio 0.2 --tes_ratio 0.4 --dataset PTB --seed 1 
+ python main.py --experiment_description exp1 --run_description run1 --data_path './data/' --selected_dataset PTB --seed 1 --training_mode self_supervised --lr 1e-6
+ python run_preprocess.py --trn_ratio 0.2 --val_ratio 0.1 --tes_ratio 0.4 --dataset PTB --seed 1 
+ python main.py --experiment_description exp1 --run_description run1 --data_path './data/' --selected_dataset PTB --seed 1 --training_mode train_linear --lr 1e-6

We suggest setting learning_rate to small values.


For research cooperation, please contact shuang.zhou@connect.polyu.hk


## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@inproceedings{ijcai2024p624,
  title     = {Denoising-Aware Contrastive Learning for Noisy Time Series},
  author    = {Zhou, Shuang and Zha, Daochen and Shen, Xiao and Huang, Xiao and Zhang, Rui and Chung, Korris},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {5644--5652},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/624},
  url       = {https://doi.org/10.24963/ijcai.2024/624},
}
```
