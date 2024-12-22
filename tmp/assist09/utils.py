
import numpy as np
from scipy import sparse
def build_adj_list():

    qs_table = sparse.load_npz('../../hy-tmp/data2009/qs_table.npz').toarray() 
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]

    uq_table = sparse.load_npz('../../hy-tmp/data2009/uq_table.npz').toarray()
    qu_table=uq_table.T 
    num_u = uq_table.shape[0]  
    num_q2 = uq_table.shape[1]  

    q_neighbors_list = [[] for _ in range(num_question)] 
    s_neighbors_list = [[] for _ in range(num_skill)] 

    q_neighbors_u_list = [[] for _ in range(num_q2)]  
    u_neighbors_list = [[] for _ in range(num_u)] 

    for q_id in range(num_question):
        s_ids = np.reshape(np.argwhere(qs_table[q_id] > 0), [-1]).tolist() 
        q_neighbors_list[q_id] += s_ids 
        u_ids = np.reshape(np.argwhere(qu_table[q_id] > 0), [-1]).tolist()  
        q_neighbors_u_list[q_id] += u_ids

        for s_id in s_ids:
            s_neighbors_list[s_id].append(q_id)
        for u_id in u_ids:
            u_neighbors_list[u_id].append(q_id)

    return q_neighbors_list, s_neighbors_list,q_neighbors_u_list,u_neighbors_list

def gen_tm_graph(q_neighbors_list, s_neighbors_list, q_neighbors_u_list,u_neighbors_list,q_neighbor_size=4, s_neighbor_size=10,q_neighbor_u_size=4,u_neighbor_size=10):

    num_question = len(q_neighbors_list)
    num_skill = len(s_neighbors_list)
    num_u = len(u_neighbors_list)
    num_q2 = len(q_neighbors_u_list)
    print(num_question, num_q2)

    q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32) 
    s_neighbors = np.zeros([num_skill, s_neighbor_size], dtype=np.int32) 
    q_neighbors_u = np.zeros([num_q2, q_neighbor_u_size], dtype=np.int32)  
    u_neighbors = np.zeros([num_u, u_neighbor_size], dtype=np.int32) 

    for i, neighbors in enumerate(q_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= q_neighbor_size: 
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0: 
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=True)
    for i, neighbors in enumerate(s_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= s_neighbor_size:
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0:
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=True)

    for i, neighbors in enumerate(q_neighbors_u_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= q_neighbor_u_size: 
            q_neighbors_u[i] = np.random.choice(neighbors, q_neighbor_u_size, replace=False)
            continue
        if len(neighbors) > 0: 
            q_neighbors_u[i] = np.random.choice(neighbors, q_neighbor_u_size, replace=True)

    for i, neighbors in enumerate(u_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= u_neighbor_size:
            u_neighbors[i] = np.random.choice(neighbors, u_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0:
            u_neighbors[i] = np.random.choice(neighbors, u_neighbor_size, replace=True)

    return q_neighbors, s_neighbors,q_neighbors_u,u_neighbors