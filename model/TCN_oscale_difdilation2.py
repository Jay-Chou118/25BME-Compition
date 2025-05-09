"""
作者：yueyue
日期：2023年11月18日
TCN 模型（tsai版本）
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Function
import torch.nn.functional as F
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class SampaddingConv1D_BN(nn.Module):
    def __init__(self,in_channels,out_channels,ks,stride,padding,dilation):
        super(SampaddingConv1D_BN, self).__init__()
        # self.padding = nn.ConstantPad1d((int((ks - 1) / 2), int(ks / 2)), 0)
        self.conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=ks,stride=stride,padding=padding,dilation=dilation)
        self.conv = weight_norm(self.conv)
        self.chomp = Chomp1d(padding)
        self.net = nn.Sequential(self.conv, self.chomp) if ks != 1 else nn.Sequential(self.conv)
    def init_weight(self):
        self.conv.weight.data.normal_(0, 0.01)
    def forward(self,x):
        x = self.net(x)
        return x
class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks_list, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()

        self.convlist1 = nn.ModuleList()
        for ks in ks_list:
            conv = SampaddingConv1D_BN(ni,nf,ks,stride,padding*(ks-1),dilation)
            conv.init_weight()
            self.convlist1.append(conv)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.convlist2 = nn.ModuleList()
        for ks in ks_list:
            conv = SampaddingConv1D_BN(nf*len(ks_list),nf,ks,stride,padding*(ks-1),dilation)
            conv.init_weight()
            self.convlist2.append(conv)
        self.relu2 = nn.ReLU()
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(ni,nf*len(ks_list),1) if ni != nf else None
    def init_weights(self):
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        conv_result_list1 = []
        for conv in self.convlist1:
            conv_result = conv(x)
            conv_result_list1.append(conv_result)
        out = self.dropout1(self.relu1(torch.cat(tuple(conv_result_list1), 1)))

        conv_result_list2 = []
        for conv in self.convlist2:
            conv_result = conv(out)
            conv_result_list2.append(conv_result)
        out = self.dropout2(self.relu2(torch.cat(tuple(conv_result_list2), 1)))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list
def TemporalConvNet(c_in, layers, ks=2, dropout=0.,conv_small=32,dilations=None):
    temp_layers = []
    receptive_field_shape = 875//conv_small
    prime_list = get_Prime_number_in_a_range(1,receptive_field_shape)
    print('primelist',prime_list)
    for i in range(len(layers)):
        dilation_size = dilations[i]
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf//len(prime_list), prime_list, stride=1, dilation=dilation_size, padding=dilation_size, dropout=dropout)]
    return nn.Sequential(*temp_layers)

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()
    def forward(self, x):
        return self.flatten(self.gap(x))
class TCN(nn.Module):
    def __init__(self, c_in, c_out, layers=8*[30], ks=7, conv_dropout=0., fc_dropout=0.,conv_small=32,dilations=None):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout, conv_small=conv_small,dilations=dilations)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)

