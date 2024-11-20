import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import argparse

import math
import pywt
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../data_szhou/', help='dataset name: Georgia')
parser.add_argument('--dataset', type=str, default='mitbih_kaggle', help='dataset name: Georgia')
parser.add_argument('--seed', type=int, default=1, help='seed value')
parser.add_argument('--trn_ratio', type=float, default=0.7, help='train data ratio')
parser.add_argument('--val_ratio', type=float, default=0.1, help='valid data ratio')
parser.add_argument('--tes_ratio', type=float, default=0.2, help='test data ratio')

parser.add_argument('--denoise', type=str, default='False', help='whether denoise')
args = parser.parse_args()


if args.dataset == 'mitbih_kaggle':
    args.data_dims = 187
if args.dataset == 'ptb_kaggle':
    args.data_dims = 187
if args.dataset == 'SVDB':
    args.data_dims = 89
if args.dataset == 'Epilepsy':
    args.data_dims = 178
if args.dataset == 'SleepEDF':
    args.data_dims = 3000
if args.dataset == 'Georgia':
    args.data_dims = 1000
if args.dataset == 'PTB':
    args.data_dims = 1000
if args.dataset == 'CPSC18':
    args.data_dims = 1000
if args.dataset == 'FaultDiagnosis':
    args.data_dims = 5120


def fill_wave_length(wave, fixed_length):
    """
    Pads or truncates a wave to a fixed length.
    :param wave: Input 1D wave array.
    :param fixed_length: Target length for the wave.
    :return: Resized wave with the specified fixed length.
    """
    if len(wave) < fixed_length:
        num_zeros = fixed_length - len(wave)
        revised_arr = np.concatenate((wave, np.zeros(num_zeros)))
    else:
        revised_arr = wave[:fixed_length]
    return revised_arr


def read_data_folder(folder_name, file_names):
    ecg_all_data = np.zeros((data_num, args.data_dims))
    for j in range(data_num):
        file_j_name = file_names[j]
        file_j_path = os.path.join(folder_name, file_j_name)
        df_j = pd.read_csv(file_j_path, sep=",", header=None)
        ecg_j = df_j.values.T
        signal_1d_arr = ecg_j[0]
        signal_1d_arr = fill_wave_length(signal_1d_arr, args.data_dims)
        ecg_all_data[j] = signal_1d_arr
    return ecg_all_data


output_dir = './data_szhou/' + args.dataset + '/'
if os.path.isdir(output_dir) == False:
    os.makedirs(output_dir)

label_file_path = args.root + args.dataset + '_label_all.csv'
df_gnd = pd.read_csv(label_file_path)
file_names = df_gnd['Recording'].values
gnd_all = df_gnd['label'].values
if min(gnd_all) == 1:
    gnd_all = np.array([i-1 for i in gnd_all])
    print('********* now the label starts from 0 *********')

data_num = file_names.shape[0]
print(file_names.shape)
print(file_names[0:5], gnd_all[0:5])

data_folder_org = args.root + args.dataset + '_szhou_all'
ecg_data_org = read_data_folder(data_folder_org+'/', file_names)
ecg_data_denoise1 = read_data_folder(data_folder_org+'_denoise1/', file_names)
ecg_data_denoise2 = read_data_folder(data_folder_org+'_denoise2/', file_names)
ecg_data_denoise3 = read_data_folder(data_folder_org+'_denoise3/', file_names)
ecg_data_denoise4 = read_data_folder(data_folder_org+'_denoise4/', file_names)
ecg_data_denoise5 = read_data_folder(data_folder_org+'_denoise5/', file_names)
ecg_data_denoise6 = read_data_folder(data_folder_org+'_denoise6/', file_names)
ecg_data_denoise7 = read_data_folder(data_folder_org+'_denoise7/', file_names)
ecg_data_denoise8 = read_data_folder(data_folder_org+'_denoise8/', file_names)
ecg_data_denoise9 = read_data_folder(data_folder_org+'_denoise9/', file_names)
ecg_data_denoise10 = read_data_folder(data_folder_org+'_denoise10/', file_names)
samples_Gau_noise = read_data_folder(data_folder_org+'_Gaussian/', file_names)
ecg_data_noisy1 = read_data_folder(data_folder_org+'_noisy1/', file_names)
ecg_data_noisy2 = read_data_folder(data_folder_org+'_noisy2/', file_names)
ecg_data_noisy3 = read_data_folder(data_folder_org+'_noisy3/', file_names)
ecg_data_noisy4 = read_data_folder(data_folder_org+'_noisy4/', file_names)
ecg_data_noisy5 = read_data_folder(data_folder_org+'_noisy5/', file_names)
ecg_data_noisy6 = read_data_folder(data_folder_org+'_noisy6/', file_names)
ecg_data_noisy7 = read_data_folder(data_folder_org+'_noisy7/', file_names)
ecg_data_noisy8 = read_data_folder(data_folder_org+'_noisy8/', file_names)
ecg_data_noisy9 = read_data_folder(data_folder_org+'_noisy9/', file_names)
ecg_data_noisy10 = read_data_folder(data_folder_org+'_noisy10/', file_names)

indices_all = np.arange(0, data_num)
train_val_ids, test_ids = train_test_split(indices_all, test_size=args.tes_ratio, random_state=args.seed)
train_ids, val_ids = train_test_split(train_val_ids, test_size=(args.val_ratio / (args.trn_ratio + args.val_ratio)), random_state=args.seed+11)
print('train size: ', train_val_ids.shape[0], train_val_ids.shape[0] / data_num)
print('valid size: ', val_ids.shape[0], val_ids.shape[0] / data_num)
print('test size: ', test_ids.shape[0], test_ids.shape[0] / data_num)

# save train data
dat_dict_trn = dict()
dat_dict_trn["samples"] = torch.from_numpy(ecg_data_org[train_ids]).unsqueeze(1)
dat_dict_trn["labels"] = torch.from_numpy(gnd_all[train_ids])
dat_dict_trn["samples_denoise1"] = torch.from_numpy(ecg_data_denoise1[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise2"] = torch.from_numpy(ecg_data_denoise2[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise3"] = torch.from_numpy(ecg_data_denoise3[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise4"] = torch.from_numpy(ecg_data_denoise4[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise5"] = torch.from_numpy(ecg_data_denoise5[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise6"] = torch.from_numpy(ecg_data_denoise6[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise7"] = torch.from_numpy(ecg_data_denoise7[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise8"] = torch.from_numpy(ecg_data_denoise8[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise9"] = torch.from_numpy(ecg_data_denoise9[train_ids]).unsqueeze(1)
dat_dict_trn["samples_denoise10"] = torch.from_numpy(ecg_data_denoise10[train_ids]).unsqueeze(1)
dat_dict_trn["samples_Gau_noise"] = torch.from_numpy(samples_Gau_noise[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy1"] = torch.from_numpy(ecg_data_noisy1[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy2"] = torch.from_numpy(ecg_data_noisy2[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy3"] = torch.from_numpy(ecg_data_noisy3[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy4"] = torch.from_numpy(ecg_data_noisy4[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy5"] = torch.from_numpy(ecg_data_noisy5[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy6"] = torch.from_numpy(ecg_data_noisy6[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy7"] = torch.from_numpy(ecg_data_noisy7[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy8"] = torch.from_numpy(ecg_data_noisy8[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy9"] = torch.from_numpy(ecg_data_noisy9[train_ids]).unsqueeze(1)
dat_dict_trn["samples_noisy10"] = torch.from_numpy(ecg_data_noisy10[train_ids]).unsqueeze(1)
torch.save(dat_dict_trn, os.path.join(output_dir, "train.pt"))

# save val data
dat_dict_val = dict()
dat_dict_val["samples"] = torch.from_numpy(ecg_data_org[val_ids]).unsqueeze(1)
dat_dict_val["labels"] = torch.from_numpy(gnd_all[val_ids])
dat_dict_val["samples_denoise1"] = torch.from_numpy(ecg_data_denoise1[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise2"] = torch.from_numpy(ecg_data_denoise2[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise3"] = torch.from_numpy(ecg_data_denoise3[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise4"] = torch.from_numpy(ecg_data_denoise4[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise5"] = torch.from_numpy(ecg_data_denoise5[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise6"] = torch.from_numpy(ecg_data_denoise6[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise7"] = torch.from_numpy(ecg_data_denoise7[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise8"] = torch.from_numpy(ecg_data_denoise8[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise9"] = torch.from_numpy(ecg_data_denoise9[val_ids]).unsqueeze(1)
dat_dict_val["samples_denoise10"] = torch.from_numpy(ecg_data_denoise10[val_ids]).unsqueeze(1)
dat_dict_val["samples_Gau_noise"] = torch.from_numpy(samples_Gau_noise[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy1"] = torch.from_numpy(ecg_data_noisy1[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy2"] = torch.from_numpy(ecg_data_noisy2[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy3"] = torch.from_numpy(ecg_data_noisy3[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy4"] = torch.from_numpy(ecg_data_noisy4[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy5"] = torch.from_numpy(ecg_data_noisy5[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy6"] = torch.from_numpy(ecg_data_noisy6[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy7"] = torch.from_numpy(ecg_data_noisy7[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy8"] = torch.from_numpy(ecg_data_noisy8[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy9"] = torch.from_numpy(ecg_data_noisy9[val_ids]).unsqueeze(1)
dat_dict_val["samples_noisy10"] = torch.from_numpy(ecg_data_noisy10[val_ids]).unsqueeze(1)
torch.save(dat_dict_val, os.path.join(output_dir, "val.pt"))

# save test data
dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(ecg_data_org[test_ids]).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(gnd_all[test_ids])
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))
