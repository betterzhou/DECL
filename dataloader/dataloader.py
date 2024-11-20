import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Load_Datasets(Dataset):
    def __init__(self, dataset, config, training_mode):
        super(Load_Datasets, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        X_noisy_Gau = dataset["samples_Gau_noise"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
            X_noisy_Gau = X_noisy_Gau.unsqueeze(2)
            for j in range(1, 11):
                dataset["samples_noisy"+str(j)] = dataset["samples_noisy"+str(j)].unsqueeze(2)
                dataset["samples_denoise" + str(j)] = dataset["samples_denoise" + str(j)].unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)
            X_noisy_Gau = X_noisy_Gau.permute(0, 2, 1)
            for j in range(1, 11):
                dataset["samples_noisy" + str(j)] = dataset["samples_noisy" + str(j)].permute(0, 2, 1)
                dataset["samples_denoise" + str(j)] = dataset["samples_denoise" + str(j)].permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train
            self.X_noisy_Gau = X_noisy_Gau

            self.X_denoise_1 = dataset["samples_denoise1"]
            self.X_denoise_2 = dataset["samples_denoise2"]
            self.X_denoise_3 = dataset["samples_denoise3"]
            self.X_denoise_4 = dataset["samples_denoise4"]
            self.X_denoise_5 = dataset["samples_denoise5"]
            self.X_denoise_6 = dataset["samples_denoise6"]
            self.X_denoise_7 = dataset["samples_denoise7"]
            self.X_denoise_8 = dataset["samples_denoise8"]
            self.X_denoise_9 = dataset["samples_denoise9"]
            self.X_denoise_10 = dataset["samples_denoise10"]
            self.X_noisy_1 = dataset["samples_noisy1"]
            self.X_noisy_2 = dataset["samples_noisy2"]
            self.X_noisy_3 = dataset["samples_noisy3"]
            self.X_noisy_4 = dataset["samples_noisy4"]
            self.X_noisy_5 = dataset["samples_noisy5"]
            self.X_noisy_6 = dataset["samples_noisy6"]
            self.X_noisy_7 = dataset["samples_noisy7"]
            self.X_noisy_8 = dataset["samples_noisy8"]
            self.X_noisy_9 = dataset["samples_noisy9"]
            self.X_noisy_10 = dataset["samples_noisy10"]
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or "fine_tune":
            return (self.x_data[index], self.y_data[index],
                    self.X_noisy_Gau[index],
                    self.X_noisy_1[index], self.X_noisy_2[index], self.X_noisy_3[index], self.X_noisy_4[index], self.X_noisy_5[index],
                    self.X_noisy_6[index], self.X_noisy_7[index], self.X_noisy_8[index], self.X_noisy_9[index], self.X_noisy_10[index],
                    self.X_denoise_1[index], self.X_denoise_2[index], self.X_denoise_3[index], self.X_denoise_4[index], self.X_denoise_5[index],
                    self.X_denoise_6[index], self.X_denoise_7[index], self.X_denoise_8[index], self.X_denoise_9[index], self.X_denoise_10[index])
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Load_Dataset_test(Dataset):
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset_test, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index], self.x_data[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    batch_size = configs.batch_size

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Datasets(train_dataset, configs, training_mode)
    valid_dataset = Load_Datasets(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset_test(test_dataset, configs, training_mode)

    if train_dataset.__len__() < batch_size:
        batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader
