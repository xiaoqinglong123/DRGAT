import csv
import torch
import random
import numpy as np
import scipy.io as sio

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def data_pro(args):
    dataset = dict()

    if args.data == 'Ldata':
    #Ldata
        dataset['md_p'] = np.loadtxt(args.dataset_path + '/Ldata/dr_dis_association_mat.txt')
        dataset['md_true'] = np.loadtxt(args.dataset_path + '/Ldata/dr_dis_association_mat.txt')
    if args.data == 'Cdata':
        #cdata
        P = sio.loadmat(args.dataset_path + '/Fdata and Cdata/Cdataset.mat')
        dataset['md_p']=P['didr'].transpose()
        dataset['md_true']=P['didr'].transpose()
    if args.data == 'Fdata':
        #Fdata
        P = sio.loadmat(args.dataset_path + '/Fdata and Cdata/Fdataset.mat')
        if args.shiyan == 0 or args.shiyan == 2:
            dataset['md_p'] = P['didr'].transpose()
            dataset['md_true'] = P['didr'].transpose()
        if args.shiyan == 1:
            dataset['md_p'] = P['didr']
            dataset['md_true'] = P['didr']
    if args.data == 'LRSSL':
        #LRSSL
        dataset['md_p'] = np.loadtxt(args.dataset_path + '/LRSSL/drug-disease.txt')
        dataset['md_true'] = np.loadtxt(args.dataset_path + '/LRSSL/drug-disease.txt')

    #自己的数据集公用的代码
    dataset['md_p']=torch.from_numpy(dataset['md_p']).float()
    dataset['md_true'] = torch.from_numpy(dataset['md_true']).float()

    zero_index = []
    one_index = []

    for i in range(dataset['md_p'].size(0)):

        for j in range(dataset['md_p'].size(1)):

            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]


    "disease sim"
    if args.data == 'LRSSL':
    # LRSSL
        dd_f_matrix = np.loadtxt(args.dataset_path + '/LRSSL/sim_disease.txt')

    # cdata 和Fdata 都可以用
    if args.data == 'Cdata':
        dd_f_matrix = P['disease']
    if args.data == 'Fdata':
        if args.shiyan == 0 or args.shiyan == 2:
            dd_f_matrix = P['disease']
        if args.shiyan == 1:
            dd_f_matrix = P['drug']
    # Ldata
    # "disease sim"
    if args.data == 'Ldata':
        dd_f_matrix = np.loadtxt(args.dataset_path + '/Ldata/dis_sim.txt')

    #下面是公用代码
    dd_f_matrix = torch.from_numpy(dd_f_matrix).float()
    dd_f_edge_index = get_edge_index(dd_f_matrix)
    dataset['dd_f'] = {'data_matrix': dd_f_matrix, 'edges': dd_f_edge_index}

    "drug sim1"
    # cdata 和Fdata 都可以用
    if args.data == 'Cdata':
        mm_f_matrix = P['drug']
    if args.data == 'Fdata':
        if args.shiyan == 0 or args.shiyan == 2:
            mm_f_matrix = P['drug']
        if args.shiyan == 1:
            mm_f_matrix = P['disease']

    #lrssl
    # "drug sim"
    # sim_drug_go,sim_drug_chemical,sim_drug_domain
    if args.data == 'LRSSL':
        mm_f_matrix = np.loadtxt(args.dataset_path + '/LRSSL/sim_drug_chemical.txt')

    #Ldata
    # pathway_sim,enzyme_sim,drug_interaction_sim,structure_sim,(LAGCN)target_sim
    if args.data == 'Ldata':
        mm_f_matrix = np.loadtxt(args.dataset_path + '/Ldata/structure_sim.txt')


    # 下面是公用代码

    mm_f_matrix = torch.from_numpy(mm_f_matrix).float()
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}


    return dataset


