from param import parameter_parser
from DRMGCN import DRMGCN
from dataprocessing import data_pro
from utils import show_auc
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scipy.io as sio
import torch.nn.functional as F


def lossF(lossType, predictions, targets):
    predictions=predictions.flatten()
    targets = targets.flatten()
    if lossType == 'cross_entropy':
        pos_weight = 1.
        neg_weight = 1.

        weightTensor = torch.zeros(len(targets))
        weightTensor = weightTensor.cuda()
        weightTensor[targets == 1] = pos_weight
        weightTensor[targets == 0] = neg_weight
        if (predictions.min() < 0) | (predictions.max() > 1):
            losses = F.binary_cross_entropy_with_logits(predictions.double(), targets.double(), weight=weightTensor)
        else:
            losses = F.binary_cross_entropy(predictions.double(), targets.double(), weight=weightTensor)
    elif lossType == 'MF_all':
        losses = torch.pow((predictions - targets), 2).mean()
    return losses

def train(model,dataset,A0,ldi, optimizer, opt,i,fold_ROC,fold_AUPR):

   
    model.train()

    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score = model(dataset)
        loss =lossF('cross_entropy', score, A0.cuda())
        loss.backward()
        optimizer.step()
        print(loss.item())
        score = score.detach().cpu().numpy()
        scoremin, scoremax = score.min(), score.max()
        score = (score - scoremin) / (scoremax - scoremin)
    return score




def crossCV(args):

    if args.data == 'Fdata':
        P = sio.loadmat(args.dataset_path + '/Fdata and Cdata/Fdataset.mat')
        ldi=P['didr'].transpose()
        
    A = torch.from_numpy(ldi).float()
    A = A.cuda()

    fold_ROC = []
    fold_AUPR = []
    
    dataset = data_pro(args)
    A0 = A.clone()

    model = DRMGCN(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    fold_ROC, fold_AUPR = train(model, dataset,A0,ldi, optimizer, args,i,fold_ROC,fold_AUPR)



if __name__ == "__main__":
    args = parameter_parser()
    crossCV(args)
