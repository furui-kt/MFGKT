"""
训练并测试模型
"""
import datetime
import time
start = time.time()
start_time = datetime.datetime.now()
print('开始时间：',start_time)
import gc
gc.disable();

import torch
import os
import math
import time
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from data_process import min_seq_len, max_seq_len
from dataset import UserDataset
from tm import TM
from params import *
from utils import gen_tm_graph, build_adj_list

gc.collect()
torch.cuda.empty_cache()

time_now =datetime.datetime.now().strftime('%Y_%m_%d#%H_%M_%S')

params = {
    'max_seq_len': max_seq_len, #200
    'min_seq_len': min_seq_len, #20
    'lr_gamma': 0.75,
    'batch_size': 32,  
    'size_q_neighbors': 4,  
    'size_s_neighbors': 10,  
    'size_u_neighbors': 10,
    'size_q_neighbors_u': 4, 
    'num_workers': 16,
    'prefetch_factor': 4,
    'agg_hops': 3,  
    'emb_dim': 100, 
    'hard_recap': False,  
    'dropout': (0.1, 0.2),
    'rank_k': 10,  
    'pre_train': True,  
    'k_fold': 5, 
    'epochs': 10, 
    'lr': 0.001, 
}
batch_size = params['batch_size']
qs_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE)  
num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
uq_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/uq_table.npz').toarray(), dtype=torch.int64, device=DEVICE)  
num_u = torch.tensor(uq_table.shape[0], device=DEVICE)
num_q2=torch.tensor(uq_table.shape[1], device=DEVICE)
qu_table=uq_table.T
q_neighbors_list, s_neighbors_list,q_neighbors_u_list,u_neighbors_list = build_adj_list()  #不加q2_neighbors_list
q_neighbors, s_neighbors,q_neighbors_u,u_neighbors= gen_tm_graph(q_neighbors_list, s_neighbors_list, q_neighbors_u_list,u_neighbors_list, params['size_q_neighbors'], params['size_s_neighbors'],params['size_q_neighbors_u'],params['size_u_neighbors'])


q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)#问题的知识邻居
s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)#知识的问题邻居
q_neighbors_u = torch.tensor(q_neighbors_u, dtype=torch.int64, device=DEVICE)#问题的学生邻居
u_neighbors = torch.tensor(u_neighbors, dtype=torch.int64, device=DEVICE)#学生的问题邻居
model = TM(
    num_question, num_skill, num_q2, num_u, 
    q_neighbors, s_neighbors, q_neighbors_u,u_neighbors,
    qs_table,uq_table,qu_table,
    agg_hops=params['agg_hops'],
    emb_dim=params['emb_dim'],
    dropout=params['dropout'],
    hard_recap=params['hard_recap'],
    pre_train=params['pre_train'],
).to(DEVICE)

loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) 
dataset = UserDataset()  
data_len = len(dataset)  
epoch_total = 0

optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
torch.optim.lr_scheduler.ExponentialLR(optimizer, params['lr_gamma'])

y_label_aver = np.zeros([3, params['epochs']])
y_label_all = np.zeros([3, params['epochs'] * params['k_fold']]) 
k_fold = KFold(n_splits=params['k_fold'] ,shuffle=True) 

for epoch in range(params['epochs']): 
    train_loss_aver = train_acc_aver = train_auc_aver = 0
    test_loss_aver = test_acc_aver = test_auc_aver = 0
    for fold, (train_indices, test_indices) in enumerate(k_fold.split(dataset)):
        train_set = Subset(dataset, train_indices)  
        test_set = Subset(dataset, test_indices)  
        if DEVICE.type == 'cpu':  
            train_loader = DataLoader(train_set, batch_size=batch_size) 
            test_loader = DataLoader(test_set, batch_size=batch_size)  #
        else:  
            train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=params['num_workers'],
                                      pin_memory=True, prefetch_factor=params['prefetch_factor'])
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=params['num_workers'],
                                     pin_memory=True, prefetch_factor=params['prefetch_factor'])
        train_data_len, test_data_len = len(train_set), len(test_set)
        time0 = time.time()
        train_step = train_loss = train_total = train_right = train_auc = 0
        for data in train_loader:
            optimizer.zero_grad()
            x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
            user_u = data[:, :, 3].to(DEVICE)
            y_hat = model(x, y_target, mask, user_u)
            y_hat = torch.masked_select(y_hat, mask)
            y_pred = torch.ge(y_hat, torch.tensor(0.5))
            y_target = torch.masked_select(y_target, mask)
            
            loss = loss_fun(y_hat, y_target.to(torch.float32))
            train_loss += loss.item()
            acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
            train_right += torch.sum(torch.eq(y_target, y_pred))
            train_total += torch.sum(mask)

            auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
            train_auc += auc * len(x) / train_data_len 
            loss.backward()
            optimizer.step()
            train_step += 1 

        train_loss, train_acc = train_loss / train_step, train_right / train_total
        train_loss_aver += train_loss 
        train_acc_aver += train_acc 
        train_auc_aver += train_auc 

        with torch.no_grad():
            test_step = test_loss = test_total = test_right = test_auc = 0
            for data in test_loader:
                j += 1
                x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
                user_u = data[:, :, 3].to(DEVICE)
                y_hat = model(x, y_target, mask,  user_u)
                y_hat = torch.masked_select(y_hat, mask.to(torch.bool))
                y_pred = torch.ge(y_hat, torch.tensor(0.5))
                y_target = torch.masked_select(y_target, mask.to(torch.bool))
                loss = loss_fun(y_hat, y_target.to(torch.float32))
                test_loss += loss.item()
          
                acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
                test_right += torch.sum(torch.eq(y_target, y_pred))
                test_total += torch.sum(mask)
                try:
                   auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
                except ValueError:
                    pass
                test_auc += auc * len(x) / test_data_len
                test_step += 1 

            test_loss, test_acc = test_loss / test_step, test_right / test_total
            test_loss_aver += test_loss 
            test_acc_aver += test_acc 
            test_auc_aver += test_auc 
            epoch_total += 1
            time1 = time.time()
            run_time = time1 - time0
            y_label_all[0][epoch_total-1], y_label_all[1][epoch_total-1], y_label_all[2][epoch_total-1] = test_loss, test_acc, test_auc

            gc.collect()
            torch.cuda.empty_cache()

    train_loss_aver /= params['k_fold'] #对每一轮 求训练集损失值的多次交叉验证的平均值
    train_acc_aver /= params['k_fold']  #对每一折求训练集acc平均值
    train_auc_aver /= params['k_fold']  #对每一折求训练集auc平均值
    test_loss_aver /= params['k_fold']  #对每一折求测试集损失值平均值
    test_acc_aver /= params['k_fold']   #对每一折求测试集acc平均值
    test_auc_aver /= params['k_fold']   #对每一折求测试集auc平均值

    print("第",(epoch+1),"轮所有交叉验证的平均值" )
    print(LOG_G + f'training: loss: {train_loss_aver:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver: .4f}' + LOG_END)
    print(LOG_G + f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}' + LOG_END)


end = time.time()
end_time = datetime.datetime.now()
print('结束时间：',end_time)

run_time=(end-start)/3600
print("train_test运行时间:%.2f小时"%run_time)

gc.enable();