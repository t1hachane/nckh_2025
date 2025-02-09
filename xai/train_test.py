""" Training and testing of the model
"""
import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
# from model import MMDynamic
from models import MMDynamic, init_model_dict
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
import copy

cuda = True if torch.cuda.is_available() else False

class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0.001):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_epoch = None
        self.early_stop = False
        self.metric_higher_better_max = 0.0
        self.delta = delta

    def __call__(self, metric_higher_better, epoch, model_dict):

        score = metric_higher_better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early stop at epoch {epoch}th after {self.counter} epochs not increasing score from epoch {self.best_epoch}th with best score {self.best_score}')
        else:
            self.best_score = score
            self.save_checkpoint(metric_higher_better, epoch, model_dict)
            self.counter = 0

    def save_checkpoint(self, metric_higher_better, epoch, model_dict):
        self.best_weights = copy.deepcopy(model_dict)
        self.metric_higher_better_max = metric_higher_better
        self.best_epoch = epoch


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)    
    return y_onehot

def prepare_trte_data(data_folder, view_list, postfix_tr='_tr', postfix_te='_val'):
    # load train, test data
    num_view = len(view_list)
    # begin: need to fix
    labels_tr = np.loadtxt(os.path.join(data_folder, f"labels{postfix_tr}.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, f"labels{postfix_te}.csv"), delimiter=',')

    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view+1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+f"{postfix_tr}.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+f"{postfix_te}.csv"), delimiter=','))
    
    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels

def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list

def train_epoch(data_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss, _ = model(data_list, label)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()


def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob

def save_checkpoint(model, checkpoint_path, filename="MMDynamic.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)

def train_test(data_folder, view_list, num_class,
               lr,
               num_epoch, 
               rseed, 
               modelpath, testonly,
               test_inverval = 50,
               postfix_tr='_tr',
               postfix_te='_val',
               patience=7,
               verbose=False,
               hidden_dim=[1000]):
    if rseed>=0:
        torch.manual_seed(rseed)
        np.random.seed(rseed)
        
    step_size = 500
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, postfix_tr, postfix_te)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_class, dim_list, hidden_dim)
    model = model_dict["MMDynamic"]
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    if testonly == True:
        load_checkpoint(model, os.path.join(modelpath, data_folder, 'MMDynamic.pt'))
        te_prob = test_epoch(data_test_list, model)
        if num_class == 2:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
        else:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    else:    
        print("\nTraining...")
        if patience is not None:
            # using early stopping: 
            early_stopping = EarlyStopping(patience = patience, verbose = verbose)

        best_model_dict = None
        for epoch in range(num_epoch+1):
            train_epoch(data_tr_list, labels_tr_tensor, model, optimizer)
            scheduler.step()
            te_prob = test_epoch(data_test_list, model)
            te_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            if verbose == 'True':
                if epoch % test_inverval == 0:
                    print("\nTest: Epoch {:d}".format(epoch))
                    if num_class == 2:
                        print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                        print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
                    else:
                        print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                        print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            if patience is not None:
                if num_class == 2:
                    score = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                else:
                    score = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')

                early_stopping(score, epoch, model_dict)
                best_model_dict = early_stopping.best_weights
                
                if early_stopping.early_stop:
                    print(f"Early stopping triggered. Using best model from epoch {early_stopping.best_epoch}")

                    model = best_model_dict 
                    break

        if early_stopping is not None:
            if early_stopping.early_stop:
                # Use the best model from early stopping
                model_dict = early_stopping.best_weights
            else:
                # No early stop => store the final epoch's model
                early_stopping.best_weights = copy.deepcopy(model_dict)
                model_dict = early_stopping.best_weights

    return model
        
        


