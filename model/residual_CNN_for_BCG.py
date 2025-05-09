import numpy as np
import torch
from torch._C import TupleType
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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, auc, roc_curve,confusion_matrix
from scipy.special import softmax
from sklearn.metrics import classification_report   # 新加的评估结果工具

# 设置随机种子
np.random.seed(666)
random.seed(666)

# model = Model(875, 64,3).cuda()

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

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
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

class plain_CNN_Model(torch.nn.Module):   # 模型
    def __init__(self, input_size, hidden_size, output_size):
        super(plain_CNN_Model, self).__init__()   # 继承父类的属性 batch*1*1250
        self.c1 = nn.Sequential(nn.Conv1d(1, 16, 100, 5),
                                nn.BatchNorm1d(16),
                                nn.ReLU())                  # batch*16*230

        self.c2 = nn.Sequential(nn.Conv1d(16,32,3,2,1),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32,32,3,1,1),
                                nn.BatchNorm1d(32),
                                nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv1d(32,64,3,2,1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Conv1d(64,64,3,1,1),
                                nn.BatchNorm1d(64),
                                nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv1d(64,128,3,2,1),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Conv1d(128,128,3,1,1),
                                nn.BatchNorm1d(128),
                                nn.ReLU())
        self.c5 = nn.Sequential(nn.Conv1d(128,256,3,2,1),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Conv1d(256,256,3,1,1),
                                nn.BatchNorm1d(256),
                                nn.ReLU())
        #self.c6 = resnet_block(256, 512, 1)

        self.c6 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # dropout
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):  # Xavier初始化
        """Xavier initialization for the fully connected layer
        """
        fcs = [self.fc1,self.fc2]
        for fc in fcs:
            r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
            fc.weight.data.uniform_(-r, r)
            fc.bias.data.fill_(0)

    def forward(self, inputs):
        inputs = torch.unsqueeze(inputs, 1)
        #print(inputs.size())
        x = self.c1(inputs)
        #print(x.size())
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        #x = self.c7(x)
        x = x.view(-1, 256)   # 确保输入全连接层的维度对应
        #print(x.size())
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.relu(x)
        return x

class res_CNN_Model(torch.nn.Module):   # 模型
    def __init__(self, input_size, hidden_size, output_size):
        super(res_CNN_Model, self).__init__()   # 继承父类的属性 batch*1*1250

        self.c1 = nn.Sequential(nn.Conv1d(1, 16, 100, 5),
                                nn.BatchNorm1d(16),
                                nn.ReLU())                  # batch*16*230

        self.c2 = resnet_block(16, 32, 1)                   # batch*64*115
        self.c3 = resnet_block(32, 64, 1)                  # batch*128*58
        self.c4 = resnet_block(64, 128, 1)                 # batch*256*29
        self.c5 = resnet_block(128, 256, 1)                 # batch*512*15
        #self.c6 = resnet_block(256, 512, 1)

        self.c6 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # dropout
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):  # Xavier初始化
        """Xavier initialization for the fully connected layer
        """
        fcs = [self.fc1,self.fc2]
        for fc in fcs:
            r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
            fc.weight.data.uniform_(-r, r)
            fc.bias.data.fill_(0)

    def forward(self, inputs):
        # inputs = torch.unsqueeze(inputs, 1)
        #print(inputs.size())
        x = self.c1(inputs)
        #print(x.size())
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        #x = self.c7(x)
        x = x.view(-1, 256)   # 确保输入全连接层的维度对应
        #print(x.size())
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.relu(x)
        return x

