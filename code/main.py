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

#损失函数
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

        #损失函数
        loss =lossF('cross_entropy', score, A0.cuda())#交叉熵损失函数

        loss.backward()
        optimizer.step()

        score = score.detach().cpu().numpy()
        scoremin, scoremax = score.min(), score.max()
        score = (score - scoremin) / (scoremax - scoremin)

        # np.savetxt('lagcnsa.txt', score, fmt='%10.5f', delimiter=',')

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

    disease = P['Wdname']  # disese
    drug = P['Wrname']


    A = torch.from_numpy(ldi).float()
    A = A.cuda()


    # 设置几折
    nfold = args.validation
    N = args.drug_number  # 行数，也就是药物的个数

    idx = np.arange(N)  # 0到663

    np.random.shuffle(idx)#放在循环外面就是正确的
    fold_ROC = []
    fold_AUPR = []

    print("---start {} Fold cross  validation---".format(nfold ))

    for i in range(nfold):

        print("Fold {}".format(i + 1))
        dataset = data_pro(args)

        #弄出训练集A0
        A0 = A.clone()
        if args.shiyan == 0:
            for j in range(i * N // nfold, (i + 1) * N // nfold):
                A0[idx[j], :] = torch.zeros(A.shape[1])

        if args.shiyan == 1:
            # 把某i列设置为0
            A0[i, :] = torch.zeros(A.shape[1])
        print(A0.shape)
        if args.shiyan == 2:
            # 第8个104300，第120个168600  已选前两个   第138个，D182280，
            print(disease[137])
            print(disease[7])
            print(disease[119])
            

        print(A0.equal(A))

        model = DRMGCN(args)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#0.01,0.001,0.0001

        #应该传入训练集
        fold_ROC, fold_AUPR = train(model, dataset,A0,ldi, optimizer, args,i,fold_ROC,fold_AUPR)


    print("average AUROC is {:.4} , average AUPR is {:.4}".format(sum(fold_ROC) / len(fold_ROC),
                                                                  sum(fold_AUPR) / len(fold_AUPR)))
    return sum(fold_ROC) / len(fold_ROC),sum(fold_AUPR) / len(fold_AUPR)



if __name__ == "__main__":
    args = parameter_parser()

    crossCV(args)


    exp_name='00'#设置调哪个,先测试打印到模型中  结果对不对

    if exp_name=='anlifenxi':


        args.validation=1
        args.shiyan = 2
        args.epoch = 1000

        averageAUROC, averageAUPR = crossCV(args)



    if exp_name=='dulishiyan':

        args.drug_number = 313
        args.disease_number = 593
        args.validation=313
        args.shiyan = 1
        args.epoch = 1000

        averageAUROC, averageAUPR = crossCV(args)

        np.savetxt('dulishiyan'  + '.txt', np.vstack((averageAUROC, averageAUPR)),
                   fmt='%10.5f', delimiter=',')

    if exp_name=='ndata':
        data_ls = ['LRSSL']#'Cdata','Ldata','LRSSL'
        for data in data_ls:
            args.data = data
            args.fm = 256
            args.fd = 256
            if data=='Cdata':
                args.drug_number = 663
                args.disease_number=409
                args.out_channels = 64
            if data=='Ldata':
                args.drug_number = 269
                args.disease_number = 598
                args.out_channels = 128
            if data=='LRSSL':
                args.drug_number = 763
                args.disease_number = 681
                args.out_channels = 128
            print(args.data)
            print(args.fm)
            print(args.out_channels)
            averageAUROC, averageAUPR = crossCV(args)

            np.savetxt('data' + str(data) + '.txt', np.vstack((averageAUROC, averageAUPR)),
                       fmt='%10.5f', delimiter=',')

    if exp_name=='lr':
        lr_ls = [0.0001]
        for lr in lr_ls:
            args.lr = lr
            print(lr)
            averageAUROC, averageAUPR = crossCV(args)
            np.savetxt('lr' + str(lr) + '.txt', np.vstack((averageAUROC, averageAUPR)),
                       fmt='%10.5f', delimiter=',')

    if exp_name=='gcnlays':
        gcnlays_ls = [5]
        for gcnlays in gcnlays_ls:
            args.gcn_layers = gcnlays
            print(gcnlays)
            averageAUROC, averageAUPR = crossCV(args)
            np.savetxt('gcnlays' + str(gcnlays) + '.txt', np.vstack((averageAUROC, averageAUPR)),
                       fmt='%10.5f', delimiter=',')

    if exp_name == 'embeddingsize':
        embeddingsize_ls = [32]
        for embeddingsize in embeddingsize_ls:
            print(embeddingsize)
            args.fm = embeddingsize
            args.fd = embeddingsize
            averageAUROC,averageAUPR=crossCV(args)
            np.savetxt('embeddingsize'+str(embeddingsize)+'.txt', np.vstack((averageAUROC,averageAUPR)), fmt='%10.5f', delimiter=',')

    if exp_name=='filternumber':
        filternumber_ls = [64]
        for filternumber in filternumber_ls:
            print(filternumber)
            args.out_channels = filternumber
            averageAUROC, averageAUPR = crossCV(args)
            np.savetxt('mid_dims' + str(filternumber) + '.txt', np.vstack((averageAUROC, averageAUPR)),
                       fmt='%10.5f', delimiter=',')





