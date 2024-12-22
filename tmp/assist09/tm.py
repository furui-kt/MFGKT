
import pandas as pd
import torch
import numpy as np
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell
from params import DEVICE  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TM(Module):
    def __init__(self, num_question, num_skill, num_q2, num_u,q_neighbors, s_neighbors, q_neighbors_u, u_neighbors,qs_table,uq_table,qu_table,
             agg_hops=3,emb_dim=100,dropout=(0.1, 0.2), hard_recap=False, rank_k=10, pre_train=True):
        super(TM, self).__init__()
        self.model_name = "MFGKT"
        self.num_question = num_question
        self.num_skill = num_skill
        self.num_q2 = num_q2
        self.num_u = num_u

        self.q_neighbors = q_neighbors
        self.s_neighbors = s_neighbors
        self.q_neighbors_u = q_neighbors_u
        self.u_neighbors = u_neighbors

        self.agg_hops = agg_hops
        self.qs_table = qs_table
        self.uq_table = uq_table
        self.qu_table = qu_table

        self.emb_dim = emb_dim
        self.hard_recap = hard_recap
        self.rank_k = rank_k
        self.pre_train = pre_train

        if pre_train:
            print('使用预训练')

            _weight_q = torch.load(f='../../hy-tmp/data2009/q_embedding.pt')
            _weight_s = torch.load(f='../../hy-tmp/data2009/s_embedding.pt')
            _weight_u = torch.load(f='../../hy-tmp/data2009/u_embedding.pt')
            _weight_q2 = torch.load(f='../../hy-tmp/data2009/q2_embedding.pt')
            np.savetxt('../../hy-tmp/data2009/s_embedding.csv',_weight_s.cpu().detach().numpy(),fmt='%.2f',delimiter=',')


        else:
            print('不使用预训练')
            self.emb_table_question = Embedding(num_question, emb_dim)
            self.emb_table_skill = Embedding(num_skill, emb_dim)
            self.emb_table_u = Embedding(num_u, emb_dim)
            self.emb_table_q2 = Embedding(num_q2, emb_dim)

        self.emb_table_response = Embedding(2, emb_dim)

    def forward(self, question, response, mask,user):
        batch_size, seq_len = question.shape
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        u_neighbor_size= self.u_neighbors.shape[1]
        q_neighbor_u_size = self.q_neighbors_u.shape[1]
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1))
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)
        for t in range(seq_len - 1):
            question_t = question[:, t]
            response_t = response[:, t]
            u_t = user[:, t]
            mask_t = torch.eq(mask[:, t],torch.tensor(1))
            emb_response_t = self.emb_table_response(response_t)

            _batch_size = len(node_neighbors[0])

            for i in range(self.agg_hops):

                neighbor_shape = [_batch_size] + [(q_neighbor_size if j % 2 == 0 else s_neighbor_size) for j in range(i + 1)]
                neighbor_u_shape = [_batch_size] + [(q_neighbor_u_size if j % 2 == 0 else u_neighbor_size) for j in range(i + 1)]

            shape = tuple(len(row) for row in node_neighbors)
            shape_u = tuple(len(row) for row in node_neighbors_u)
            emb_node_neighbor = []

            emb_node_neighbor_u = []

            for i, nodes in enumerate(node_neighbors):
                if i % 2 == 0:
                    emb_node_neighbor.append(self.emb_table_question(nodes))
                else:
                    emb_node_neighbor.append(self.emb_table_skill(nodes))

            for i, nodes in enumerate(node_neighbors_u):
                if i % 2 == 0:
                    emb_node_neighbor_u.append(self.emb_table_q2(nodes))
                else:
                    emb_node_neighbor_u.append(self.emb_table_u(nodes))

            shape = tuple(len(row) for row in emb_node_neighbor)
            shape_u = tuple(len(row) for row in emb_node_neighbor_u)
            emb0_question_t = self.aggregate(emb_node_neighbor)
            emb_question_t = torch.zeros(batch_size, self.emb_dim, device=DEVICE)
            emb_question_t[mask_t] = emb0_question_t
            emb_question_t[~mask_t] = self.emb_table_question(question_t[~mask_t])


            q_next = question[:, t + 1]
            skills_related = self.qs_table[q_next]
            u_related = self.qu_table[q_next]
            skills_related_list = []
            u_related_list = []
            max_num_skill = 1
            max_num_u = 1

            for i in range(batch_size):
                skills_index = torch.nonzero(skills_related[i]).squeeze()
                u_index = torch.nonzero(u_related[i]).squeeze()

                if len(skills_index.shape) == 0:
                    skills_related_list.append(
                        torch.unsqueeze(self.emb_table_skill(skills_index), dim=0))
                else:
                    skills_related_list.append(self.emb_table_skill(skills_index))
                    if skills_index.shape[0] > max_num_skill:
                        max_num_skill = skills_index.shape[0]
                if len(u_index.shape) == 0:
                    u_related_list.append(
                        torch.unsqueeze(self.emb_table_u(u_index), dim=0))
                else:
                    u_related_list.append(self.emb_table_u(u_index))
                    if u_index.shape[0] > max_num_u:
                        max_num_u = u_index.shape[0]

            emb_q_next = self.emb_table_question(q_next)
            emb_q_next_u = self.emb_table_q2(q_next)
            qs_concat = torch.zeros(batch_size, max_num_skill + 1, self.emb_dim).to(DEVICE)
            qu_concat = torch.zeros(batch_size, max_num_u + 1, self.emb_dim).to(DEVICE)

            for i, emb_skills in enumerate(skills_related_list):
                num_qs = 1 + emb_skills.shape[0]
                emb_next = torch.unsqueeze(emb_q_next[i], dim=0)
                qs_concat[i, 0: num_qs] = torch.cat((emb_next, emb_skills), dim=0)



            if t == 0:
                y_hat[:, 0] = 0.5
                y_hat[:, 1] = self.predict(qsu_concat, torch.unsqueeze(lstm_output,dim=1))
                continue
            if self.hard_recap:
                history_time = self.recap_hard(q_next, question[:, 0:t])
                selected_states = []
                max_num_states = 1
                for row, selected_time in enumerate(history_time):
                    current_state = torch.unsqueeze(lstm_output[row], dim=0)
                    if len(selected_time) == 0:
                        selected_states.append(current_state)
                    else:
                        selected_state = state_history[row, torch.tensor(selected_time, dtype=torch.int64)]
                        selected_states.append(torch.cat((current_state, selected_state), dim=0))
                        if (selected_state.shape[0] + 1) > max_num_states:
                            max_num_states = selected_state.shape[0] + 1
                current_history_state = torch.zeros(batch_size, max_num_states, self.emb_dim).to(DEVICE)

                for b, c_h_state in enumerate(selected_states):
                    num_states = c_h_state.shape[0]
                    current_history_state[b, 0: num_states] = c_h_state
            else:
                current_state = lstm_output.unsqueeze(dim=1)
                if t <= self.rank_k:
                    current_history_state = torch.cat((current_state, state_history[:, 0:t]), dim=1)
                else:
                    Q = self.emb_table_question(q_next).clone().detach().unsqueeze(dim=-1)
                    K = self.emb_table_question(question[:, 0:t]).clone().detach()
                    product_score = torch.bmm(K, Q).squeeze(dim=-1)
                    _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                    select_history = torch.cat(tuple(state_history[i][indices[i]].unsqueeze(dim=0)
                                                     for i in range(batch_size)), dim=0)
                    current_history_state = torch.cat((current_state, select_history), dim=1)


        return y_hat


    def recap_hard(self, q_next, q_history):
        batch_size = q_next.shape[0]
        q_neighbor_size, s_neighbor_size = self.q_neighbors.shape[1], self.s_neighbors.shape[1]
        q_next = q_next.reshape(-1)
        skill_related = self.q_neighbors[q_next].reshape((batch_size, q_neighbor_size)).reshape(
            -1)
        q_related = self.s_neighbors[skill_related].reshape((batch_size, q_neighbor_size * s_neighbor_size)).tolist()
        time_select = [[] for _ in range(batch_size)]
        for row in range(batch_size):
            key = q_history[row].tolist()
            query = q_related[row]
            for t, k in enumerate(key):
                if k in query:
                    time_select[row].append(t)
        return time_select

    def recap_soft(self, rank_k=10):
        pass
