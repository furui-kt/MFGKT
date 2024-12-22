"""
预训练
"""
import math
import numpy as np
import torch
torch.set_printoptions(threshold=np.inf) 
import torch.nn as nn
from scipy import sparse
from ptm import PTM
from params import DEVICE
from torch.utils.tensorboard import SummaryWriter
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch, gc
gc.collect()
torch.cuda.empty_cache()

qs_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE)
qq_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/qq_table.npz').toarray(), dtype=torch.int64, device=DEVICE) 
ss_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/ss_table.npz').toarray(), dtype=torch.int64, device=DEVICE) 
uq_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/uq_table.npz').toarray(), dtype=torch.int64, device=DEVICE) 
uu_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/uu_table.npz').toarray(), dtype=torch.int64, device=DEVICE) 
qq2_table = torch.tensor(sparse.load_npz('../../hy-tmp/data2009/qq2_table.npz').toarray(), dtype=torch.int64, device=DEVICE)

num_q = qs_table.shape[0] 
num_s = qs_table.shape[1] 
batch_size =256
num_batch = math.ceil(num_q / batch_size)  
num_u = uq_table.shape[0] 
model = PTM(qs_table, qq_table, ss_table,uq_table, uu_table,qq2_table,emb_dim=100)
model = model.to(DEVICE)

print('开始预训练模型')

optimizer = torch.optim.Adam(params=model.parameters())   

mse_loss = [nn.MSELoss().to(DEVICE) for _ in range(6)]   

for epoch in range(20):
    print(epoch) 
    train_loss = 0  
    for idx_batch in range(num_batch): 
        optimizer.zero_grad()

        idx_start = idx_batch * batch_size 
        idx_end = min((idx_batch + 1) * batch_size, num_q)  

        q_embedding = model.q_embedding[idx_start: idx_end]  
        s_embedding = model.s_embedding  
        u_embedding = model.u_embedding 
        q2_embedding = model.q2_embedding[idx_start: idx_end] 
       
        qs_target = model.qs_target[idx_start: idx_end]  
        qq_target = model.qq_target[idx_start: idx_end, idx_start: idx_end]  
        ss_target = model.ss_target  
        uq_target = model.uq_target[:,idx_start: idx_end]  
        uu_target = model.uu_target  

        qq2_target = model.qq2_target[idx_start: idx_end, idx_start: idx_end]
        qs_logit, qq_logit, ss_logit, uq_logit, uu_logit, qq2_logit = model.forward(q_embedding, s_embedding,u_embedding,q2_embedding)  

        loss_qs = torch.sqrt(mse_loss[0](qs_logit, qs_target)).sum() 

        loss_qq = torch.sqrt(mse_loss[1](qq_logit, qq_target)).sum()  
        loss_ss = torch.sqrt(mse_loss[2](ss_logit, ss_target)).sum()  
        loss_uq = torch.sqrt(mse_loss[3](uq_logit, uq_target)).sum()  
        loss_uu = torch.sqrt(mse_loss[4](uu_logit, uu_target)).sum()  
        loss_qq2 = torch.sqrt(mse_loss[5](qq2_logit, qq2_target)).sum()  

        loss = loss_qs + loss_qq + loss_qq + loss_uq + loss_uu  + loss_qq2 
        train_loss += loss.item()
        loss.backward()  
        optimizer.step()

    print(f'----------epoch: {epoch + 1}, 总损失train_loss: {train_loss}') 
    print(f'----------各损失loss_qs: {loss_qs}, loss_qq: {loss_qs}, loss_qq: {loss_qq}, loss_ss: {loss_ss}, loss_uq: {loss_uq}, loss_uu: {loss_uu}, loss_qq2: {loss_qq2}')

torch.save(model.q_embedding, f='../../hy-tmp/data2009/q_embedding.pt') 
torch.save(model.s_embedding, f='../../hy-tmp/data2009/s_embedding.pt') 
torch.save(model.u_embedding, f='../../hy-tmp/data2009/u_embedding.pt') 
torch.save(model.q2_embedding, f='../../hy-tmp/data2009/q2_embedding.pt') 



