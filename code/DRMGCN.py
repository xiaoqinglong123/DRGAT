import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import conv
torch.backends.cudnn.enabled = False


class DRMGCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(DRMGCN, self).__init__()
        self.args = args
        #drug1
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)#嵌入size，两层gcn
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x3_f = GCNConv(self.args.fm, self.args.fm)  # 嵌入size，两层gcn
        self.gcn_x4_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x5_f = GCNConv(self.args.fm, self.args.fm)



        #disease1
        self.gcn_y1_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y3_f = GCNConv(self.args.fd, self.args.fd)  # 嵌入size，两层gcn
        self.gcn_y4_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y5_f = GCNConv(self.args.fd, self.args.fd)



        self.globalAvgPool_x = nn.MaxPool2d((self.args.fm, self.args.drug_number), (1, 1))  # 最大池化层
        self.globalAvgPool_y = nn.MaxPool2d((self.args.fd, self.args.disease_number), (1, 1))

        self.fc1_x = nn.Linear(in_features=self.args.drview*self.args.gcn_layers,
                             out_features=5*self.args.drview*self.args.gcn_layers)

        self.fc2_x = nn.Linear(in_features=5*self.args.drview*self.args.gcn_layers,
                             out_features=self.args.drview*self.args.gcn_layers)

        self.fc1_y = nn.Linear(in_features=self.args.diview * self.args.gcn_layers,
                             out_features=5 * self.args.diview * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.diview * self.args.gcn_layers,
                             out_features=self.args.diview * self.args.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv1d(in_channels=self.args.drview*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.args.diview*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):

        # print(self.args.out_channels)
        # print(self.args.fm)
        # print(self.args.fd)
        # print(self.args.gcn_layers)

        torch.manual_seed(1)
        x_m = torch.randn(self.args.drug_number, self.args.fm)
        x_d = torch.randn(self.args.disease_number, self.args.fd)

        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f3 = torch.relu(self.gcn_x3_f(x_m_f2, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f4 = torch.relu(self.gcn_x4_f(x_m_f3, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f5 = torch.relu(self.gcn_x5_f(x_m_f4, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))



        if self.args.gcn_layers==1:
            XM= x_m_f1
            #消融实验只用第二个
            # XM = x_m_f2
        if self.args.gcn_layers == 2:
            XM = torch.cat((x_m_f1, x_m_f2), 1).t()  # [1024, 593]
        if self.args.gcn_layers==3:
            XM = torch.cat((x_m_f1, x_m_f2,x_m_f3), 1).t()  # [1024, 593]
        if self.args.gcn_layers==4:
            XM = torch.cat((x_m_f1, x_m_f2,x_m_f3,x_m_f4), 1).t()  # [1024, 593]
        if self.args.gcn_layers==5:
            XM = torch.cat((x_m_f1, x_m_f2,x_m_f3,x_m_f4,x_m_f5), 1).t()  # [1024, 593]

        XM = XM.view(1, self.args.drview*self.args.gcn_layers, self.args.fm, -1)#[1, 2, 512, 593]

        x_channel_attenttion = self.globalAvgPool_x(XM)#[1, 2, 1, 1]
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)#[1, 2]
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)#[1, 10]
        x_channel_attenttion = torch.relu(x_channel_attenttion)#[1, 10]
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)#[1, 2]
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)#[1, 2]
        #[1, 2, 1, 1]
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        #[1, 2, 1, 1]
        XM_channel_attention = x_channel_attenttion * XM
        # [1, 2, 1, 1]
        XM_channel_attention = torch.relu(XM_channel_attention)#[1, 2, 512, 593]
        # cnn_x自定义的卷积
        x = self.cnn_x(XM_channel_attention)#[1, 128, 1, 593]
        x = x.view(self.args.out_channels, self.args.drug_number).t()#[593, 128]

        ##------------------操作一模一样-----------
        ##dd1两次卷积
        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f3 = torch.relu(self.gcn_y3_f(y_d_f2, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f4 = torch.relu(self.gcn_y4_f(y_d_f3, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f5 = torch.relu(self.gcn_y5_f(y_d_f4, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))



        if self.args.gcn_layers==1:
            YD= y_d_f1
            # 消融实验只用第二个
            # YD= y_d_f2
        if self.args.gcn_layers == 2:
            YD = torch.cat((y_d_f1, y_d_f2), 1).t()
        if self.args.gcn_layers==3:
            YD =torch.cat((y_d_f1, y_d_f2,y_d_f3), 1).t()  # [1024, 593]
        if self.args.gcn_layers==4:
            YD = torch.cat((y_d_f1, y_d_f2,y_d_f3,y_d_f4), 1).t()  # [1024, 593]
        if self.args.gcn_layers==5:
            YD = torch.cat((y_d_f1, y_d_f2,y_d_f3,y_d_f4,y_d_f5), 1).t()  # [1024, 593]



        YD = YD.view(1, self.args.diview*self.args.gcn_layers, self.args.fd, -1)

        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        YD_channel_attention = y_channel_attenttion * YD

        YD_channel_attention = torch.relu(YD_channel_attention)

        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.out_channels, self.args.disease_number).t()

        return x.mm(y.t())

# y_d_f1gat = torch.relu(self.gat_y1_f(data['dd_f']['data_matrix'].cuda(), data['dd_f']['edges'].cuda()))
        # y_d_f2gat = torch.relu(self.gat_y2_f(y_d_f1gat.cuda(), data['dd_f']['edges'].cuda()))

# x_m_f1gat = torch.relu(self.gat_x1_f(data['mm_f']['data_matrix'].cuda(),data['mm_f']['edges'].cuda()))
        # x_m_f2gat = torch.relu(self.gat_x2_f(x_m_f1gat.cuda(),data['mm_f']['edges'].cuda()))
#拉普拉斯
        #alpha1 = torch.randn(self.args.out_channels, self.args.disease_number).double()
        # alpha2 = torch.randn(self.args.out_channels,self.args.miRNA_number).double()
        #
        # x = laplacian(x)  # [495, 495]
        # y = laplacian(y)  # [383, 383]
        #
        # out1 = torch.mm(x, alpha1)  # [495, 383]
        # out2 = torch.mm(y, alpha2)  # [383, 495]
        #
        # out = (out1 + out2.T) / 2
def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = torch.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = torch.where(t.isinf(D_5), torch.full_like(D_5, 0), D_5)
    L_D_11 = torch.mm(D_5, L_D_1)
    L_D_11 = torch.mm(L_D_11, D_5)
    return L_D_11






