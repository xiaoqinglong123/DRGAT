import torch
import random
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

    if args.data == 'Fdata':
        P = sio.loadmat(args.dataset_path + '/Fdata and Cdata/Fdataset.mat')
        dataset['md_p'] = P['didr'].transpose()
        dataset['md_true'] = P['didr'].transpose()

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
    if args.data == 'Fdata':
        dd_f_matrix = P['disease']

    dd_f_matrix = torch.from_numpy(dd_f_matrix).float()
    dd_f_edge_index = get_edge_index(dd_f_matrix)
    dataset['dd_f'] = {'data_matrix': dd_f_matrix, 'edges': dd_f_edge_index}

    "drug sim"
    if args.data == 'Fdata':
        mm_f_matrix = P['drug']

    mm_f_matrix = torch.from_numpy(mm_f_matrix).float()
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}


    return dataset
