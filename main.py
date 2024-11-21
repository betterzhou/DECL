import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model
from trainer.trainer import Trainer, model_test
from utils import copy_Files, cal_f1s_naive
from sklearn.metrics import roc_auc_score
from utils import _logger, set_requires_grad
import pandas as pd
from config_files.mitbih_kaggle_Configs import Config
import warnings
warnings.filterwarnings("ignore")

start_time = datetime.now()
parser = argparse.ArgumentParser()


# ####################### Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description',     default='HAR_experiments',  type=str,   help='Experiment Description')
parser.add_argument('--run_description',            default='test1',            type=str,   help='Experiment Description')
parser.add_argument('--seed',                       default=0,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str)
parser.add_argument('--selected_dataset',           default='HAR',              type=str)
parser.add_argument('--data_path',                  default=r'./data_szhou/',           type=str,   help='Path containing dataset')
parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='cuda:0',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')
parser.add_argument('--test_results_path',          default='./results/',           type=str,   help='Path containing dataset')
parser.add_argument('--num_epoch', type=int, default=100, help='seed value')
parser.add_argument('--lr', type=float, default=1e-6, help='')
parser.add_argument('--lambda1', type=float, default=0.6, help='hyper-params')
parser.add_argument('--lambda2', type=float, default=0.7, help='hyper-params')

args = parser.parse_args()
if args.selected_dataset == 'mitbih_kaggle':
    print('**************************** load mitbih_kaggle_Configs ...\n\n')
if args.selected_dataset == 'ptb_kaggle':
    from config_files.ptb_kaggle_Configs import Config
    print('**************************** load ptb_kaggle_Configs ...\n\n')
if args.selected_dataset == 'SVDB':
    from config_files.SVDB_Configs import Config
    print('**************************** load SVDB_Configs ...\n\n')
if args.selected_dataset == 'Epilepsy':
    from config_files.Epilepsy_Configs import Config
    print('**************************** load Epilepsy_Configs ...\n\n')
if args.selected_dataset == 'SleepEDF':
    from config_files.SleepEDF_Configs import Config
    print('**************************** load sleepEDF_Configs ...\n\n')
if args.selected_dataset == 'Georgia':
    from config_files.Georgia_Configs import Config
    print('**************************** load Georgia_Configs ...\n\n')
if args.selected_dataset == 'PTB':
    from config_files.PTB_Configs import Config
    print('**************************** load PTB_Configs ...\n\n')
if args.selected_dataset == 'CPSC18':
    from config_files.CPSC18_Configs import Config
    print('**************************** load CPSC18_Configs ...\n\n')
if args.selected_dataset == 'FaultDiagnosis':
    from config_files.FaultDiagnosis_Configs import Config
    print('**************************** load sleepEDF_Configs ...\n\n')


device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Config()
args.num_classes = configs.num_classes

SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0
# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

data_path = os.path.join(args.data_path, data_type)
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)


if "train_linear" in training_mode:
    load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         "saved_models"))

    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)


model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=args.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
if training_mode == "self_supervised":
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode, args)

if training_mode != "self_supervised":
    outs = model_test(model, temporal_contr_model, test_dl, device, training_mode, dataset_class_dim=args.num_classes)
    total_loss, total_acc, pred_labels, true_labels, probs = outs
    # -------------------------------------------------------------------------
    y_trues = true_labels.astype(int)
    y_pred = pred_labels.astype(int)
    f1_classes, f1_micro, f1_all, report_df = cal_f1s_naive(y_trues, y_pred, configs)
    avg_f1 = np.mean(f1_classes)
    f1s = f1_all.tolist()
    if probs.shape[1] == 2:
        print('binary-class classification, input shall be 1d array')
        auc_macro = roc_auc_score(y_trues, probs[:, 1], average="macro", multi_class='ovr')
        auc_weighted = roc_auc_score(y_trues, probs[:, 1], average="weighted", multi_class='ovr')
    else:
        print('Multi-class classification, input shall be Nd array')
        auc_macro = roc_auc_score(y_trues, probs, average="macro", multi_class='ovr')
        auc_weighted = roc_auc_score(y_trues, probs, average="weighted", multi_class='ovr')
    f1s.append(auc_macro)
    f1s.append(auc_weighted)
    all_metric = np.array([f1s])
    colunms = report_df.columns.tolist()
    colunms.extend(['AUC_macro', 'AUC_weighted'])
    report_df = pd.DataFrame(all_metric, columns=colunms)
    report_df.to_csv(args.test_results_path + args.selected_dataset + '_' + str(args.training_mode) +
                     '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '.csv')
    # -------------------------------------------------------------------------
logger.debug(f"Training time is : {datetime.now() - start_time}")
