"""
作者：yueyue
日期：2023年11月13日
"""
import torch
from torch import nn
class singlemodel(nn.Module):
    def __init__(self,input_size,num_classes):
        super(singlemodel, self).__init__()
        self.linear = nn.Linear(input_size,num_classes)
    def forward(self,x):
        x = torch.flatten(x,1)
        return self.linear(x)

class secondlayer(nn.Module):
    def __init__(self,input_size,num_classes):
        super(secondlayer, self).__init__()
        self.linear1 = nn.Linear(input_size,512)
        self.linear2 = nn.Linear(512,num_classes)
    def forward(self,x):
        x = torch.flatten(x, 1)
        return self.linear2(self.linear1(x))