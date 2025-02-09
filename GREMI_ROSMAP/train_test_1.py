import os
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import time
import torch
import sklearn.metrics
from model_GREMI import *

# Env
from utils import *

def prepare_trte_data():
    raise NotImplementedError
def train_test(data_folder, 
               view_list, num_class,
               input_in_dim, input_hidden_dim,
                input_dropout, input_lr, input_weight_decay,
                num_epochs, save_path, step_size=500, gamma=0.2, postfix_tr='_tr',
               postfix_te='_val'):
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, postfix_tr, postfix_te)
    
    data_tr, tr_omic, tr_labels, data_te, te_omic, te_labels, exp_adj1, exp_adj2, exp_adj3 = prepare_trte_data(data_folder, view_list)
    tr_dataset = torch.utils.data.TensorDataset(tr_omic, tr_labels)
    tr_data_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=32, shuffle=True)
    te_dataset = torch.utils.data.TensorDataset(te_omic, te_labels)
    te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=32, shuffle=False)
    num_views = len(view_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = Fusion(num_class=num_class, num_views=num_views, hidden_dim=input_hidden_dim, dropout=input_dropout, in_dim=input_in_dim)

    optimizer = torch.optim.Adam(network.parameters(), lr=input_lr, weight_decay=input_weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    best_epoch = 0
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []

    for epoch in range(0, num_epochs):
        # Print epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        network.train()
        current_loss = 0.0
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        for i, data in enumerate(tr_data_loader, 0):
            batch_x, targets = data
            batch_x1 = batch_x[:, 0:200].reshape(-1, 200, 1)
            batch_x2 = batch_x[:, 200:400].reshape(-1, 200, 1)
            batch_x3 = batch_x[:, 400:600].reshape(-1, 200, 1)

            batch_x1 = batch_x1.to(torch.float32)
            batch_x2 = batch_x2.to(torch.float32)
            batch_x3 = batch_x3.to(torch.float32)
            targets = targets.long()
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_x3 = batch_x3.to(device)
            targets = targets.to(device)
            exp_adj1 = exp_adj1.to(device)
            exp_adj2 = exp_adj2.to(device)
            exp_adj3 = exp_adj3.to(device)

            optimizer.zero_grad()
            loss_fusion, tr_logits, gat_output1, gat_output2, gat_output3, output1, output2, output3 = network(batch_x1, batch_x2, batch_x3, exp_adj1, exp_adj2, exp_adj3, targets)
            tr_prob = F.softmax(tr_logits, dim=1)
            tr_pre_lab = torch.argmax(tr_prob, 1)

            loss = loss_fusion
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_corrects += torch.sum(tr_pre_lab == targets.data)
            train_num += batch_x1.size(0)

        network.eval()
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        for i, data in enumerate(te_data_loader, 0):
            batch_x, targets = data
            batch_x1 = batch_x[:, 0:200].reshape(-1, 200, 1)
            batch_x2 = batch_x[:, 200:400].reshape(-1, 200, 1)
            batch_x3 = batch_x[:, 400:].reshape(-1, 200, 1)
            batch_x1 = batch_x1.to(torch.float32)
            batch_x2 = batch_x2.to(torch.float32)
            batch_x3 = batch_x3.to(torch.float32)
            targets = targets.long()
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_x3 = batch_x3.to(device)
            targets = targets.to(device)
            exp_adj1 = exp_adj1.to(device)
            exp_adj2 = exp_adj2.to(device)
            exp_adj3 = exp_adj3.to(device)

            te_logits = network.infer(batch_x1, batch_x2, batch_x3, exp_adj1, exp_adj2, exp_adj3)
            te_prob = F.softmax(te_logits, dim=1)
            te_pre_lab = torch.argmax(te_prob, 1)

            test_corrects += torch.sum(te_pre_lab == targets.data)
            test_num += batch_x1.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        print('{} Train Loss : {:.8f} Train ACC : {:.8f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{}  Test ACC : {:.8f}'.format(epoch, test_acc_all[-1]))

        if test_acc_all[-1] > best_acc:
            best_acc = test_acc_all[-1]
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(network.state_dict())
            # Saving the model
            state = {
                'net': best_model_wts,
            }
            torch.save(state, save_path)

