
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell
from params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PTM(nn.Module):

      def __init__(self, qs_table, qq_table, ss_table,uq_table,uu_table,qq2_table, emb_dim=100):
        super().__init__()
        num_q, num_s = qs_table.shape[0], qs_table.shape[1]
        num_u = uq_table.shape[0]
        num_q2 = uq_table.shape[1]

        self.q_embedding = nn.Parameter(torch.randn(size=[num_q, emb_dim])).to(device)
        self.s_embedding = nn.Parameter(torch.randn(size=[num_s, emb_dim])).to(device)
        self.u_embedding = nn.Parameter(torch.randn(size=[num_u, emb_dim])).to(device)
        self.q2_embedding = nn.Parameter(torch.randn(size=[num_q2, emb_dim])).to(device)


        self.relu_q = nn.ReLU()
        self.relu_s = nn.ReLU()
        self.relu_u = nn.ReLU()
        self.relu_q2 = nn.ReLU()

        self.sigmoid = [nn.Sigmoid().to(device) for _ in range(6)]
        print('ptm model built') #ptm模型

      def forward(self, q_embedding, s_embedding,u_embedding,q2_embedding):
        q_embedding_fc = self.relu_q(self.fc_q(q_embedding))
        s_embedding_fc = self.relu_s(self.fc_s(s_embedding))
        u_embedding_fc = self.relu_u(self.fc_u(u_embedding))
        q2_embedding_fc = self.relu_q2(self.fc_u(q2_embedding))

        qs_logit = self.sigmoid[0](torch.mm(q_embedding_fc, s_embedding_fc.T))
        qq_logit = self.sigmoid[1](torch.mm(q_embedding_fc, q_embedding_fc.T))
        ss_logit = self.sigmoid[2](torch.mm(s_embedding_fc, s_embedding_fc.T))
        uq_logit = self.sigmoid[3](torch.mm(u_embedding_fc, q_embedding_fc.T))
        uu_logit = self.sigmoid[4](torch.mm(u_embedding_fc, u_embedding_fc.T))
        qq2_logit = self.sigmoid[5](torch.mm(q2_embedding_fc, q2_embedding_fc.T))

        return qs_logit, qq_logit, ss_logit, uq_logit, uu_logit, qq2_logit