#预处理
import pandas as pd
import numpy as np
import os
from scipy import sparse

min_seq_len = 3 
max_seq_len = 200 
if __name__ == '__main__':
    data = pd.read_csv(filepath_or_buffer='../../hy-tmp/data2009/assist09_origin346860.csv', encoding="ISO-8859-1",low_memory=False)
    data = data.sort_values(by='user_id', ascending=True)
    print('按用户id排序完成')
    data = data.drop(data[data['skill_id'] == 'NA'].index) 
    data = data.dropna(subset=['skill_id'])

    is_valid_user = data.groupby('user_id').size() >= min_seq_len
    data = data[data['user_id'].isin(is_valid_user[is_valid_user].index)]
    print('数据行清洗完成')

    data = data.loc[:, ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id', 'skill_name',
                        'ms_first_response', 'answer_type', 'attempt_count']] 
    print('数据列清洗完成')
    print(data.shape)

    num_answer = data.shape[0] 
    users = set()
    questions = set()
    skills = set()

    for row in data.itertuples(index=False):
        users.add(row[1])     
        questions.add(row[2]) 
        if isinstance(row[4], (int, float)):
            skills.add(int(row[4]))  
        else:
            skill_add = set(int(s) for s in row[4].split('_'))
            skills = skills.union(skill_add)

    data.to_csv('../../hy-tmp/data2009/assist09_processed.csv', sep=',', index=False)#data目前是241212行, 9列

    num_q = len(questions)
    num_s = len(skills)
    num_u = len(users)
    print('num_q,num_s,num_u:',num_q,num_s,num_u) 

    if not os.path.exists('../../hy-tmp/data2009/question2idx.npy'):
        questions = list(questions)
        skills = list(skills)
        users = list(users)

        question2idx = {questions[i]: i + 1 for i in range(num_q)} 
        question2idx[0] = 0 
        skill2idx = {skills[i]: i for i in range(num_s)}
        user2idx = {users[i]: i for i in range(num_u)}
        num_q += 1
        print(num_q, num_s, num_u)
        idx2question = {question2idx[q]: q for q in question2idx}
        idx2skill = {skill2idx[s]: s for s in skill2idx}
        idx2user = {user2idx[u]: u for u in user2idx}

        np.save('../../hy-tmp/data2009/question2idx.npy', question2idx)
        np.save('../../hy-tmp/data2009/skill2idx.npy', skill2idx)
        np.save('../../hy-tmp/data2009/user2idx.npy', user2idx)
        np.save('../../hy-tmp/data2009/idx2question.npy', idx2question)
        np.save('../../hy-tmp/data2009/idx2skill.npy', idx2skill)
        np.save('../../hy-tmp/data2009/idx2user.npy', idx2user)

    else:
        question2idx = np.load('../../hy-tmp/data2009/question2idx.npy', allow_pickle=True).item()
        skill2idx = np.load('../../hy-tmp/data2009/skill2idx.npy', allow_pickle=True).item()
        user2idx = np.load('../../hy-tmp/data2009/user2idx.npy', allow_pickle=True).item()
        idx2question = np.load('../../hy-tmp/data2009/idx2question.npy', allow_pickle=True).item()
        idx2skill = np.load('../../hy-tmp/data2009/idx2skill.npy', allow_pickle=True).item()
        idx2user = np.load('../../hy-tmp/data2009/idx2user.npy', allow_pickle=True).item()

    if not os.path.exists('../../hy-tmp/data2009/qs_table.npz'):
        qs_table = np.zeros([num_q, num_s], dtype=int) 
        q_set = data['problem_id'].drop_duplicates() 
        q_samples = pd.concat([data[data['problem_id'] == q_id].sample(1) for q_id in q_set])
        for row in q_samples.itertuples(index=False):
            if isinstance(row[4], (int, float)):
                qs_table[question2idx[row[2]], skill2idx[int(row[4])]] = 1
            else: 
                skill_add = [int(s) for s in row[4].split('_')] 
                for s in skill_add:
                    qs_table[question2idx[row[2]], skill2idx[s]] = 1
        print('问题-知识矩阵qs_table构建完成')


        qq_table = np.dot(qs_table, qs_table.T)  
        ss_table = np.dot(qs_table.T, qs_table) 
        print('三个邻接矩阵qs_table、qq_table、ss_table全部构建完成')
        qs_table = sparse.coo_matrix(qs_table)  
        qq_table = sparse.coo_matrix(qq_table)
        ss_table = sparse.coo_matrix(ss_table)

        sparse.save_npz('../../hy-tmp/data2009/qs_table.npz', qs_table)
        sparse.save_npz('../../hy-tmp/data2009/qq_table.npz', qq_table)
        sparse.save_npz('../../hy-tmp/data2009/ss_table.npz', ss_table)

    else:
        qs_table = sparse.load_npz('../../hy-tmp/data2009/qs_table.npz').toarray() #将稀疏矩阵转为数组
        qq_table = sparse.load_npz('../../hy-tmp/data2009/qq_table.npz').toarray()
        ss_table = sparse.load_npz('../../hy-tmp/data2009/ss_table.npz').toarray()

    if not os.path.exists('../../hy-tmp/data2009/uq_table.npz'):
        uq_table = np.zeros([num_u, num_q], dtype=int)  #
        u_set = data['user_id'].drop_duplicates() 
        u_samples = pd.concat([data[data['user_id'] == u_id].sample(1) for u_id in u_set])
        for row_u in u_samples.itertuples(index=False):
            if isinstance(row_u[2], (int, float)): 
                uq_table[user2idx[row_u[1]], question2idx[int(row_u[2])]] = 1
        uu_table = np.dot(uq_table, uq_table.T)  
        qq2_table = np.dot(uq_table.T,uq_table) 
        uq_table = sparse.coo_matrix(uq_table)
        uu_table = sparse.coo_matrix(uu_table)
        qq2_table = sparse.coo_matrix(qq2_table)

        sparse.save_npz('../../hy-tmp/data2009/uq_table.npz', uq_table)
        sparse.save_npz('../../hy-tmp/data2009/uu_table.npz', uu_table)
        sparse.save_npz('../../hy-tmp/data2009/qq2_table.npz', qq2_table)

    else:
        uq_table = sparse.load_npz('../../hy-tmp/data2009/uq_table.npz').toarray()
        uu_table = sparse.load_npz('../../hy-tmp/data2009/uu_table.npz').toarray()
        qq2_table = sparse.load_npz('../../hy-tmp/data2009/qq2_table.npz').toarray()

    if not os.path.exists('../../hy-tmp/data2009/user_seq.npy'):
        user_seq = np.zeros([num_u, max_seq_len]) 
        user_res = np.zeros([num_u, max_seq_len]) 
        num_seq = [0 for _ in range(num_u)] 
        user_mask = np.zeros([num_u, max_seq_len])
        for row in data.itertuples(index=False):
            user_id = user2idx[row[1]] 
            if num_seq[user_id] < max_seq_len - 1: 
                user_seq[user_id, num_seq[user_id]] = question2idx[row[2]]
                user_res[user_id, num_seq[user_id]] = row[3]
                user_mask[user_id, num_seq[user_id]] = 1
                num_seq[user_id] += 1
        np.save('../../hy-tmp/data2009/user_seq.npy', user_seq) 
        np.save('../../hy-tmp/data2009/user_res.npy', user_res) 
        np.save('../../hy-tmp/data2009/user_mask.npy', user_mask) 

    if not os.path.exists('../../hy-tmp/data2009/user_u.npy'):
        num_seq = [0 for _ in range(num_u)]  
        user_u = np.zeros([num_u, max_seq_len])  
        for row in data.itertuples(index=False):
            user_id = user2idx[row[1]]  
            if num_seq[user_id] < max_seq_len - 1: 
                user_u[user_id, num_seq[user_id]] = user2idx[row[1]] 
                num_seq[user_id] += 1 
        np.save('../../hy-tmp/data2009/user_u.npy', user_u) #输入数据

    print('数据全部保存至本地')