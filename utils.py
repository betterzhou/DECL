import logging
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score, roc_auc_score
from torch import nn


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

def cal_f1s_naive(y_trues, y_scores, args):
    report = classification_report(y_trues, y_scores, output_dict=True)
    report_df = pd.DataFrame(report)
    class_colunms = [str(i) for i in range(args.num_classes)]
    classes_f1 = report_df.loc[['f1-score'], class_colunms]
    classes_f1 = classes_f1.values[0]
    f1_all = report_df.loc[['f1-score'], :]
    f1_all = f1_all.values[0]
    f1_micro = report_df.loc[['f1-score'], ['accuracy']].values[0][0]

    return classes_f1, f1_micro, f1_all, report_df


def cal_aucs(y_trues, y_scores):
    return roc_auc_score(y_trues, y_scores, average=None, multi_class='ovr')


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


from shutil import copy


def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))
