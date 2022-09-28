from param import parameter_parser
from DRGAT import DRGAT
from dataprocessing import data_pro
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

    return losses

def train(model,dataset,traindata,ldi, optimizer, opt):

    model.train()

    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score = model(dataset)
        loss =lossF('cross_entropy', score, traindata.cuda())
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

    dataset = data_pro(args)
    traindata = A.clone()

    model = DRGAT(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, dataset,traindata,ldi, optimizer, args)

if __name__ == "__main__":
    args = parameter_parser()
    crossCV(args)
