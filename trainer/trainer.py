import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.loss import decl_triplet_loss


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, args):
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    trn_loss_epoch_list = []
    val_loss_epoch_list = []
    for epoch in range(1, args.num_epoch + 1):
        train_loss, train_acc = model_training(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                                  criterion, train_dl, config, device, training_mode, args, epoch)
        valid_loss, valid_acc, _, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode, dataset_class_dim=args.num_classes)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        trn_loss_epoch_list.append(train_loss)
        val_loss_epoch_list.append(valid_loss)

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")


def model_training(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                      device, training_mode, args, epoch):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, Gau_nsy,
                    nsy1, nsy2, nsy3, nsy4, nsy5, nsy6, nsy7, nsy8, nsy9, nsy10,
                    densy1, densy2, densy3, densy4, densy5, densy6, densy7, densy8, densy9, densy10) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            Gau_nsy = Gau_nsy.float().to(device)
            densy1, densy2, densy3, densy4, densy5= densy1.float().to(device), densy2.float().to(device), densy3.float().to(device), densy4.float().to(device), densy5.float().to(device)
            densy6, densy7, densy8, densy9, densy10= densy6.float().to(device), densy7.float().to(device), densy8.float().to(device), densy9.float().to(device), densy10.float().to(device)
            nsy1, nsy2, nsy3, nsy4, nsy5 = nsy1.float().to(device), nsy2.float().to(device), nsy3.float().to(device), nsy4.float().to(device), nsy5.float().to(device)
            nsy6, nsy7, nsy8, nsy9, nsy10 = nsy6.float().to(device), nsy7.float().to(device), nsy8.float().to(device), nsy9.float().to(device), nsy10.float().to(device)

            _, feature = model(data)
            _, features_Gau_nsy = model(Gau_nsy)
            _, features_densy1 = model(densy1)
            _, features_densy2 = model(densy2)
            _, features_densy3 = model(densy3)
            _, features_densy4 = model(densy4)
            _, features_densy5 = model(densy5)
            _, features_densy6 = model(densy6)
            _, features_densy7 = model(densy7)
            _, features_densy8 = model(densy8)
            _, features_densy9 = model(densy9)
            _, features_densy10 = model(densy10)
            _, features_nsy1 = model(nsy1)
            _, features_nsy2 = model(nsy2)
            _, features_nsy3 = model(nsy3)
            _, features_nsy4 = model(nsy4)
            _, features_nsy5 = model(nsy5)
            _, features_nsy6 = model(nsy6)
            _, features_nsy7 = model(nsy7)
            _, features_nsy8 = model(nsy8)
            _, features_nsy9 = model(nsy9)
            _, features_nsy10 = model(nsy10)

            feature = F.normalize(feature, dim=1)
            features_Gau_nsy = F.normalize(features_Gau_nsy, dim=1)
            features_densy1 = F.normalize(features_densy1, dim=1)
            features_densy2 = F.normalize(features_densy2, dim=1)
            features_densy3 = F.normalize(features_densy3, dim=1)
            features_densy4 = F.normalize(features_densy4, dim=1)
            features_densy5 = F.normalize(features_densy5, dim=1)
            features_densy6 = F.normalize(features_densy6, dim=1)
            features_densy7 = F.normalize(features_densy7, dim=1)
            features_densy8 = F.normalize(features_densy8, dim=1)
            features_densy9 = F.normalize(features_densy9, dim=1)
            features_densy10 = F.normalize(features_densy10, dim=1)

            temp_cont_loss0, _ = temporal_contr_model(feature, feature, epoch)
            temp_cont_loss_noisy, temp_cont_feat1 = temporal_contr_model(features_Gau_nsy, features_Gau_nsy, epoch)
            densy1_err, _ = temporal_contr_model(features_densy1, features_densy1, epoch)
            densy2_err, _ = temporal_contr_model(features_densy2, features_densy2, epoch)
            densy3_err, _ = temporal_contr_model(features_densy3, features_densy3, epoch)
            densy4_err, _ = temporal_contr_model(features_densy4, features_densy4, epoch)
            densy5_err, _ = temporal_contr_model(features_densy5, features_densy5, epoch)
            densy6_err, _ = temporal_contr_model(features_densy6, features_densy6, epoch)
            densy7_err, _ = temporal_contr_model(features_densy7, features_densy7, epoch)
            densy8_err, _ = temporal_contr_model(features_densy8, features_densy8, epoch)
            densy9_err, _ = temporal_contr_model(features_densy9, features_densy9, epoch)
            densy10_err, _ = temporal_contr_model(features_densy10, features_densy10, epoch)
            densy1_error = densy1_err.detach().cpu().numpy()
            densy2_error = densy2_err.detach().cpu().numpy()
            densy3_error = densy3_err.detach().cpu().numpy()
            densy4_error = densy4_err.detach().cpu().numpy()
            densy5_error = densy5_err.detach().cpu().numpy()
            densy6_error = densy6_err.detach().cpu().numpy()
            densy7_error = densy7_err.detach().cpu().numpy()
            densy8_error = densy8_err.detach().cpu().numpy()
            densy9_error = densy9_err.detach().cpu().numpy()
            densy10_error = densy10_err.detach().cpu().numpy()
            lambda1 = args.lambda1

            loss_denoise1 = decl_triplet_loss(features_densy1, feature, features_nsy1)
            loss_denoise2 = decl_triplet_loss(features_densy2, feature, features_nsy2)
            loss_denoise3 = decl_triplet_loss(features_densy3, feature, features_nsy3)
            loss_denoise4 = decl_triplet_loss(features_densy4, feature, features_nsy4)
            loss_denoise5 = decl_triplet_loss(features_densy5, feature, features_nsy5)
            loss_denoise6 = decl_triplet_loss(features_densy6, feature, features_nsy6)
            loss_denoise7 = decl_triplet_loss(features_densy7, feature, features_nsy7)
            loss_denoise8 = decl_triplet_loss(features_densy8, feature, features_nsy8)
            loss_denoise9 = decl_triplet_loss(features_densy9, feature, features_nsy9)
            loss_denoise10 = decl_triplet_loss(features_densy10, feature, features_nsy10)
            denominator = (1/densy1_error + 1/densy2_error + 1/densy3_error + 1/densy4_error + 1/densy5_error +
                           1/densy6_error + 1/densy7_error + 1/densy8_error + 1/densy9_error + 1/densy10_error)
            w1 = (1/densy1_error) / denominator
            w2 = (1/densy2_error) / denominator
            w3 = (1/densy3_error) / denominator
            w4 = (1/densy4_error) / denominator
            w5 = (1/densy5_error) / denominator
            w6 = (1/densy6_error) / denominator
            w7 = (1/densy7_error) / denominator
            w8 = (1/densy8_error) / denominator
            w9 = (1/densy9_error) / denominator
            w10 = (1 / densy10_error) / denominator

            denoise_objective = (w1 * loss_denoise1 + w2 * loss_denoise2 + w3 * loss_denoise3 + w4 * loss_denoise4 + w5 * loss_denoise5 +
                                 w6 * loss_denoise6 + w7 * loss_denoise7 + w8 * loss_denoise8 + w9 * loss_denoise9 + w10 * loss_denoise10)
            loss = (temp_cont_loss0 + temp_cont_loss_noisy) / 2 * lambda1 + denoise_objective
        else:
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()
    total_loss = torch.tensor(total_loss).mean()
    if (training_mode == "self_supervised"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, dataset_class_dim):
    model.eval()
    temporal_contr_model.eval()
    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    probs = np.zeros((1, dataset_class_dim))

    with torch.no_grad():
        for data, labels, _,   _, _, _, _, _, _, _, _, _, _,    _, _, _, _, _, _, _, _, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                predictions = F.softmax(predictions)
                probs = np.append(probs, predictions.cpu().numpy(), axis=0)
    probs = probs[1:]
    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], [], []
    else:
        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
        return total_loss, total_acc, outs, trgs, probs


def model_test(model, temporal_contr_model, test_dl, device, training_mode, dataset_class_dim):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    probs = np.zeros((1, dataset_class_dim))

    with torch.no_grad():
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                predictions = F.softmax(predictions)
                probs = np.append(probs, predictions.cpu().numpy(), axis=0)

    probs = probs[1:]
    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], [], []
    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs, probs
