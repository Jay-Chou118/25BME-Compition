###导入需要的模块
import numpy as np
import torch
from torch.nn.modules import dropout
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import argparse
import shutil
import random
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, auc, roc_curve, confusion_matrix
from scipy.special import softmax
from sklearn.metrics import classification_report   # 新加的评估结果工具
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.nn import init

# 设置随机种子
np.random.seed(42)
random.seed(42)


# model = MACNN(in_channels=1, channels=16, num_classes=3, block_num=None).cuda()

class Residual(nn.Module):  # 残差块
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.dropout(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

class Model(torch.nn.Module):   # 模型
    def __init__(self):
        super(Model, self).__init__()   # 继承父类的属性 batch*1*1250

    
        self.c1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm1d(64), 
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.c2 = resnet_block(64, 64, 2,first_block=True)   
        self.se1 = SEAttention1d(64,16)                                            
        self.c3 = resnet_block(64, 128, 2)   
        self.se2 = SEAttention1d(128,16)               
        self.c4 = resnet_block(128, 256, 2) 
        self.se3 = SEAttention1d(256,16)                
        self.c5 = resnet_block(256, 512, 2)    
        self.se4 = SEAttention1d(512,16)             
        self.c6 = nn.LSTM(input_size=40,hidden_size=64,num_layers=1,bias=True,batch_first=True) ##


        self.fc1 = nn.Linear(64,3)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)  # dropout
        self.init_weights()

    def init_weights(self):  # Xavier初始化
        """Xavier initialization for the fully connected layer
        """
        fcs = [self.fc1]
        for fc in fcs:
            r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
            fc.weight.data.uniform_(-r, r)
            fc.bias.data.fill_(0)

    def forward(self, inputs):
        inputs = torch.unsqueeze(inputs, 1)
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.se1(x)
        x = self.c3(x)
        x = self.se2(x)
        x = self.c4(x)
        x = self.se3(x)
        x = self.c5(x)
        x = self.se4(x)

        x,_ = self.c6(x)
        x = x[:,-1,:]
        x = x.view(-1,64)
        x = self.fc1(x)
        return x

class SEAttention1d(nn.Module):
    '''
    Modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    '''
    def __init__(self, channel, reduction):
        super(SEAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)

        return x*y.expand_as(x)


class macnn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, reduction=4):
        super(macnn_block, self).__init__()

        if kernel_size is None:
            kernel_size = [3,6,12]

        self.reduction = reduction

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size[1], stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size[2], stride=1, padding='same')

        self.bn = nn.BatchNorm1d(out_channels*3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        #self.se = SEAttention1d(out_channels*3,reduction=reduction)

    def forward(self,x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x_con = torch.cat([x1,x2,x3], dim=1)

        out = self.bn(x_con)
        out = self.relu(out)
        #out = self.dropout(out)

        #out_se = self.se(out)

        return out

class MACNN(nn.Module):

    def __init__(self, in_channels=1, channels=64, num_classes=3, block_num=None):
        super(MACNN, self).__init__()

        if block_num is None:
            block_num = [1, 1, 1, 1]

        self.in_channel = in_channels
        self.num_classes = num_classes
        self.channel = channels

        self.max_pool1 = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        self.max_pool4 = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        #self.max_pool5 = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.channel*24, num_classes)
        

        self.layer1 = self._make_layer(macnn_block, block_num[0], self.channel)
        self.layer2 = self._make_layer(macnn_block, block_num[1], self.channel*2)
        self.layer3 = self._make_layer(macnn_block, block_num[2], self.channel*4)
        self.layer4 = self._make_layer(macnn_block, block_num[3], self.channel*8)
        #self.layer5 = self._make_layer(macnn_block, block_num[4], self.channel*16)
        #self.layer6 = self._make_layer(macnn_block, block_num[5], self.channel*32)

    def _make_layer(self, block, block_num, channel, reduction=16):

        layers = []
        for i in range(block_num):
            layers.append(block(self.in_channel, channel, kernel_size=None,
                                stride=1, reduction=reduction))
            self.in_channel = 3*channel

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        out1 = self.layer1(x)
        out1 = self.max_pool1(out1)

        out2 = self.layer2(out1)
        out2 = self.max_pool2(out2)

        out3 = self.layer3(out2)
        out3 = self.max_pool3(out3)
        
        out4 = self.layer4(out3)
        out4 = self.avg_pool(out4)

        #out5 = self.layer5(out4)
        #out5 = self.avg_pool(out5)

        #out6 = self.layer6(out5)
        #out6 = self.avg_pool(out6)

        out = torch.flatten(out4, 1)
        out = self.fc(out)

        return out



