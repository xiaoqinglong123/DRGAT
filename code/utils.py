import numpy as np
import torch
import argparse
import scipy.io as sio
from sklearn.preprocessing import minmax_scale,scale
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score


def show_auc(ymat,ldi):

    y_true = ldi.flatten()
    ymat = ymat.flatten()

    fpr,tpr,rocth = roc_curve(y_true,ymat)
    auroc = auc(fpr,tpr)

    precision, recall, prth = precision_recall_curve(y_true, ymat)
    aupr = auc(recall, precision)
    # f1score=f1_score(y_true, ymat, average='binary')

    #画图
    # np.savetxt('roc88.txt',np.vstack((fpr,tpr)),fmt='%10.5f',delimiter=',')
    # rocdata = np.loadtxt('roc88.txt', delimiter=',')
    np.savetxt('pr8995.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')
    prdata = np.loadtxt('pr8995.txt',delimiter=',')
    plt.figure()
    # plt.plot(rocdata[0],rocdata[1])
    plt.plot(prdata[0],prdata[1])
    plt.show()

    return auroc,aupr





