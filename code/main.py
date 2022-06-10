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

    i=i+1
    model.train()
    roc = []
    pr = []

    roc_max = 0
    pr_max = 0

    if args.shiyan ==1:
        ldi = ldi[i - 1, :]

    for epoch in range(0, opt.epoch):
        model.zero_grad()

        score = model(dataset)

        loss =lossF('cross_entropy', score, A0.cuda())

        loss.backward()
        optimizer.step()

        score = score.detach().cpu().numpy()
        scoremin, scoremax = score.min(), score.max()
        score = (score - scoremin) / (scoremax - scoremin)

        if args.shiyan == 1:
            score=score[i-1, :]

        auroc, aupr = show_auc(score,ldi)

        if epoch % 1 == 0 and epoch != 0:
            print('Epoch %d |AUROC= %.4f | AUPR= %.4f' % (epoch,auroc, aupr))


        roc.append(auroc)
        pr.append(aupr)

        if roc_max < roc[epoch]:
            roc_max = roc[epoch]
            mae=epoch
        if pr_max < pr[epoch]:
            pr_max = pr[epoch]
            mare = epoch
        if epoch + 1 == opt.epoch:
            fold_ROC.append(roc_max)
            fold_AUPR.append(pr_max)

            print("this is {} fold ,the max ROC is {:.4f},and max AUPR is {:.4f}".format(i, roc_max,pr_max))
            print("the max ROC is in {} epoch,and max AUPR is in {} epoch ".format(mae, mare ))


    return fold_ROC, fold_AUPR




def crossCV(args):



    if args.data == 'Ldata':
        ldi = np.loadtxt('C:\\ky\\project\\MMGCN-main\\MMGCN-main\\datasets\\Ldata\\dr_dis_association_mat.txt')
    if args.data == 'LRSSL':
        ldi = np.loadtxt('C:\\ky\\project\\NIMGSA-main\\NIMGSA-main\\LRSSL\\drug-disease.txt')
    if args.data == 'Cdata':
        print("--------------")
        P = sio.loadmat('C:\\ky\\project\\NIMGSA-main\\NIMGSA-main\\Fdata and Cdata\\Cdataset.mat')
        ldi = P['didr'].transpose()
    if args.data == 'Fdata':
        P = sio.loadmat(args.dataset_path + '/Fdata and Cdata/Fdataset.mat')
        if args.shiyan == 0 or args.shiyan == 2:
            ldi=P['didr'].transpose()
        if args.shiyan == 1:
            ldi = P['didr']

    

    A = torch.from_numpy(ldi).float()
    A = A.cuda()


    nfold = args.validation
    N = args.drug_number  

    idx = np.arange(N) 

    np.random.shuffle(idx)
    fold_ROC = []
    fold_AUPR = []

    print("---start {} Fold cross  validation---".format(nfold ))

    for i in range(nfold):

        print("Fold {}".format(i + 1))
        dataset = data_pro(args)

        A0 = A.clone()
        
        for j in range(i * N // nfold, (i + 1) * N // nfold):
            A0[idx[j], :] = torch.zeros(A.shape[1])

    

        model = DRMGCN(args)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        fold_ROC, fold_AUPR = train(model, dataset,A0,ldi, optimizer, args,i,fold_ROC,fold_AUPR)


    print("average AUROC is {:.4} , average AUPR is {:.4}".format(sum(fold_ROC) / len(fold_ROC),
                                                                  sum(fold_AUPR) / len(fold_AUPR)))
    return sum(fold_ROC) / len(fold_ROC),sum(fold_AUPR) / len(fold_AUPR)



if __name__ == "__main__":
    args = parameter_parser()

    crossCV(args)



