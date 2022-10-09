import torch
from torch import nn
from torch_geometric.nn import conv
torch.backends.cudnn.enabled = False
import torch.nn.functional as f


class DRGAT(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(DRGAT, self).__init__()
        self.args = args

        self.gat_x1 = conv.GATConv(self.args.fm, self.args.fm,heads=self.args.heads)
        self.gat_x2 = conv.GATConv(self.args.fm, self.args.fm,heads=self.args.heads)
        self.gat_x3 = conv.GATConv(self.args.fm, self.args.fm,heads=self.args.heads)

        self.gat_y1 = conv.GATConv(self.args.fd, self.args.fd,heads=self.args.heads)
        self.gat_y2 = conv.GATConv(self.args.fd, self.args.fd,heads=self.args.heads)
        self.gat_y3 = conv.GATConv(self.args.fd, self.args.fd,heads=self.args.heads)

        self.cnn_x = nn.Conv1d(in_channels=self.args.gat_layers,
                               out_channels=self.args.emd,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.args.gat_layers,
                               out_channels=self.args.emd,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):


        torch.manual_seed(1)
        x_m = torch.randn(self.args.drug_number, self.args.fm)
        x_d = torch.randn(self.args.disease_number, self.args.fd)


        XM1 = torch.relu(self.gat_x1(x_m.cuda(), data['mm_f']['edges'].cuda()))
        XM2 = torch.relu(self.gat_x2(XM1.cuda(), data['mm_f']['edges'].cuda()))
        XM3 = torch.relu(self.gat_x3(XM2.cuda(), data['mm_f']['edges'].cuda()))

        if self.args.gat_layers==1:
            XM = XM1
        if self.args.gat_layers==2:
            XM = torch.cat((XM1, XM2), 1).t()
        if self.args.gat_layers==3:
            XM = torch.cat((XM1, XM2,XM3), 1).t()

        XM = XM.view(1, self.args.gat_layers, self.args.fm, -1)#[1, 2, 512, 593]

        x = self.cnn_x(XM)#[1, 128, 1, 593]
        x = x.view(self.args.emd, self.args.drug_number).t()#[593, 128]




        YD1 = torch.relu(self.gat_y1(x_d.cuda(), data['dd_f']['edges'].cuda()))
        YD2 = torch.relu(self.gat_y2(YD1.cuda(), data['dd_f']['edges'].cuda()))
        YD3 = torch.relu(self.gat_y3(YD2.cuda(), data['dd_f']['edges'].cuda()))


        if self.args.gat_layers==1:
            YD = YD1
        if self.args.gat_layers==2:
            YD = torch.cat((YD1, YD2), 1).t()
        if self.args.gat_layers==3:
            YD = torch.cat((YD1, YD2,YD3), 1).t()

        YD = YD.view(1, self.args.gat_layers, self.args.fd, -1)

        y = self.cnn_y(YD)
        y = y.view(self.args.emd, self.args.disease_number).t()


        return x.mm(y.t())
