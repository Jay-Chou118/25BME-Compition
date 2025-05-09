"""
作者：yueyue
日期：2023年10月30日
"""
from torchsampler import ImbalancedDatasetSampler
import numpy as np
import torch
import math
import random
import scipy.io as scio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils import data
from model.rocket_functions import *
from torch.utils.data import DataLoader, random_split
pig_infoset = [1, 2, 3, 5, 6, 7, 17, 18, 19, 20, 21, 25, 26, 28, 30, 31, 32, 36, 37, 40, 41, 44, 45]  # all pigs
pig_train_set = [1, 2, 3, 5, 6, 7, 17, 18, 19, 20, 21, 25, 26, 28, 30, 31, 32, 36, 37, 40, 41, 44,
                 45]  # train and valid pigs
pig_dic = {1:0, 2:1, 3:2, 5:3, 6:4, 7:5, 17:6, 18:7, 19:8, 20:9, 21:10, 25:11, 26:12, 28:13, 30:14, 31:15, 32:16, 36:17, 37:18, 40:19, 41:20, 44:21, 45:22}
class Bcgdataset(Dataset):
    '''读取数据集'''

    def __init__(self, root,acf_pick):
        super(Bcgdataset, self).__init__()
        #   root 数据根目录
        self.root = root
        self.acf_pick = acf_pick
        self.kernels = generate_kernels(875,10000)
    def __getitem__(self, pig_info):
        #   pig_info which pig
        # 读取这只pig的MA数据
        if pig_info == 18:
            SR_path = self.root + 'SR/SR' + str(pig_info) + '.mat'
            VF_path = self.root + 'VF/VF' + str(pig_info) + '.mat'
            data_SR = scio.loadmat(SR_path)
            data_VF = scio.loadmat(VF_path)
            # scaler = scale()
            # scaler.fit(data_SR['SR'])
            data_bcg_SR = torch.FloatTensor(data_SR['SR'])
            # scaler.fit(data_VF['VF'])
            data_bcg_VF = torch.FloatTensor(data_VF['VF'])
            data_bcg = torch.cat([data_bcg_SR.T, data_bcg_VF.T], 0)
            label = torch.cat([torch.ones(data_bcg_SR.shape[1]), torch.zeros(data_bcg_VF.shape[1])], 0)
            domain_label = torch.ones(len(label)) * pig_dic[pig_info]
        elif pig_info == 20:
            MA_path = self.root + 'MA/MA' + str(pig_info) + '.mat'
            VF_path = self.root + 'VF/VF' + str(pig_info) + '.mat'
            data_MA = scio.loadmat(MA_path)
            data_VF = scio.loadmat(VF_path)
            # scaler = scale()
            # scaler.fit(data_MA['MA'])
            data_bcg_MA = torch.FloatTensor(data_MA['MA'])
            # scaler.fit(data_VF['VF'])
            data_bcg_VF = torch.FloatTensor(data_VF['VF'])
            data_bcg = torch.cat([data_bcg_MA.T, data_bcg_VF.T], 0)
            label = torch.cat([2 * torch.ones(data_bcg_MA.shape[1]), torch.zeros(data_bcg_VF.shape[1])], 0)
            domain_label = torch.ones(len(label)) * pig_dic[pig_info]
        else:
            assert (pig_info in pig_infoset), "没有这只猪的bcg"
            MA_path = self.root + 'MA/MA' + str(pig_info) + '.mat'
            SR_path = self.root + 'SR/SR' + str(pig_info) + '.mat'
            VF_path = self.root + 'VF/VF' + str(pig_info) + '.mat'
            data_SR = scio.loadmat(SR_path)
            data_MA = scio.loadmat(MA_path)
            data_VF = scio.loadmat(VF_path)
            # scaler = scale()
            # scaler.fit(data_SR['SR'])
            data_bcg_SR = torch.FloatTensor(data_SR['SR'])
            # scaler.fit(data_MA['MA'])
            data_bcg_MA = torch.FloatTensor(data_MA['MA'])
            # scaler.fit(data_VF['VF'])
            data_bcg_VF = torch.FloatTensor(data_VF['VF'])
            data_bcg = torch.cat([data_bcg_SR.T, data_bcg_MA.T, data_bcg_VF.T], 0)
            label = torch.cat([torch.ones(data_bcg_SR.shape[1]), 2 * torch.ones(data_bcg_MA.shape[1]),
                               torch.zeros(data_bcg_VF.shape[1])], 0)
            domain_label = torch.ones(len(label)) * pig_dic[pig_info]
        if self.acf_pick == 1 or self.acf_pick == 2:
            # print('use_acf')
            data_bcg = np.array(data_bcg)
            normalize = transforms.Normalize(mean=[0],std=[1])
            data_bcg = torch.FloatTensor(data_bcg).unsqueeze(1)
            data_bcg = np.array(normalize(data_bcg).squeeze(1)).astype(np.float64)
            data_bcg = apply_kernels(data_bcg, self.kernels)
        if self.acf_pick == 1:
            return torch.FloatTensor(data_bcg).reshape(data_bcg.shape[0],2,10000), torch.LongTensor(label.numpy()),torch.LongTensor(domain_label.numpy())
        if self.acf_pick == 2:
            return torch.FloatTensor(data_bcg).reshape(data_bcg.shape[0],10000,2), torch.LongTensor(label.numpy()),torch.LongTensor(domain_label.numpy())


def cal_acf(x):
    """ 计算自相关序列 """
    x_unbiased = x - x.mean()
    x_sigma = np.sum(x_unbiased**2)
    acf = np.correlate(x_unbiased, x_unbiased, 'full') / x_sigma
    return acf[acf.size//2:]
def get_bcg_class(labels):
    '''获取标签对应的bcg类型'''
    text_labels = ['VF', 'SR', 'MA']
    return [text_labels[int(i)] for i in labels]

# 随机选择20个数据集训练，3个数据集测试
def get_random_K_fold_crossvalidation_data(dataset, pigset=pig_train_set, K=11):
    assert K > 1, '没有1折交叉验证望周知'
    X_train, y_train = None, None
    X_Valid, y_valid = None, None
    len_pigset = len(pigset)  # 23
    random.shuffle(pigset)
    slice_size = len_pigset // K  # 2
    for j in range(K):
        # 最后一折剩下的全拿走
        if j == K - 1:  # 10
            idx = slice(j * slice_size, len_pigset)  # 20-22
            pig_idx_set = pigset[idx]
            # print('K-1idx', idx)
            # print('pig', pig_idx_set)
            for pig_idx in pig_idx_set:
                x_part, y_part = dataset[pig_idx]
                if X_Valid is None:
                    X_Valid, y_valid = x_part, y_part
                else:
                    X_Valid = torch.cat([X_Valid, x_part], 0)
                    y_valid = torch.cat([y_valid, y_part], 0)
        else:
            idx = slice(j * slice_size, (j + 1) * slice_size)
            # print('idx', idx)
            pig_idx_set = pigset[idx]
            # print('pig',pig_idx_set)
            for pig_idx in pig_idx_set:
                x_part, y_part = dataset[pig_idx]
                if X_train is None:
                    X_train, y_train = x_part, y_part
                else:
                    X_train = torch.cat([X_train, x_part], 0)
                    y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_Valid, y_valid


# 留1法验证（每次仅留一只猪）
def get_loso_fold_crossvalidation_data(dataset, pigset=pig_train_set, i=0):
    K = len(pigset)
    assert K > 1, '没有1折交叉验证望周知'
    print(f'留1法{K}折交叉验证之第{i}折开始')
    X_train, y_train = None, None
    X_Valid, y_valid = None, None
    for j in range(K):
        # print('idx', idx)
        pig_idx_set = pigset[j]
        x_part, y_part = dataset[pig_idx_set]
        if j == i:
            X_Valid, y_valid = x_part, y_part
            print(f'作验证集的猪猪是{pig_idx_set}')
        elif X_train is None:
            X_train, y_train = x_part, y_part
        else:
            X_train = torch.cat([X_train, x_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_Valid, y_valid

def get_K_fold_crossvalidation_data(dataset, pigset, K=6, i=0):
    '''K折交叉验证之第i折'''
    assert K > 1, '没有1折交叉验证望周知'
    print(f'{K}折交叉验证之第{i}折开始')
    X_train, y_train, domain_y_train = None, None, None
    X_Valid, y_valid, domain_y_valid = None, None, None
    len_pigset = len(pigset)  # 23
    print(len_pigset)
    # random.shuffle(pigset)
    slice_size = math.ceil(len_pigset /K)  # 将数字“向上取整”,23 // 6 = 3.xxx = 4
    print('slice_size', slice_size)
    print(pigset)
    for j in range(K):
        # 最后一折剩下的全拿走
        if j == i:  # 0-5
            idx = slice(j * slice_size, min((j + 1) * slice_size,len_pigset))
            pig_idx_set = pigset[idx]
            print('验证集', pig_idx_set)
            # print('K-1idx', idx)
            # print('pig', pig_idx_set)
            for pig_idx in pig_idx_set:
                x_part, y_part,domain_y_part = dataset[pig_idx]
                if X_Valid is None:
                    X_Valid, y_valid, domain_y_valid = x_part, y_part, domain_y_part
                else:
                    X_Valid = torch.cat([X_Valid, x_part], 0)
                    y_valid = torch.cat([y_valid, y_part], 0)
                    domain_y_valid = torch.cat([domain_y_valid, domain_y_part], 0)
        else:
            idx = slice(j * slice_size, min((j + 1) * slice_size,len_pigset))
            pig_idx_set = pigset[idx]
            print('训练集',pig_idx_set)
            for pig_idx in pig_idx_set:
                x_part, y_part,domain_y_part = dataset[pig_idx]
                if X_train is None:
                    X_train, y_train, domain_y_train = x_part, y_part,domain_y_part
                else:
                    X_train = torch.cat([X_train, x_part], 0)
                    y_train = torch.cat([y_train, y_part], 0)
                    domain_y_train = torch.cat([domain_y_train,domain_y_part],0)
    return X_train, y_train,domain_y_train, X_Valid, y_valid,domain_y_valid

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    if is_train:
        return data.DataLoader(dataset, sampler=ImbalancedDatasetSampler(dataset), batch_size=batch_size,drop_last=True)
    else:
        return data.DataLoader(dataset, batch_size=batch_size,shuffle=is_train)
def get_domain_train_and_test(dataset,pigset):
    X_train, y_train, domain_y_train = None, None, None
    random.shuffle(pigset)
    for pig_idx in pigset:
        x_part, y_part, domain_y_part = dataset[pig_idx]
        if X_train is None:
            X_train, y_train, domain_y_train = x_part, y_part, domain_y_part
        else:
            X_train = torch.cat([X_train, x_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
            domain_y_train = torch.cat([domain_y_train, domain_y_part], 0)
    ratio = 0.8
    l = len(y_train)
    s =int(ratio*l)
    sample_indices = torch.randperm(X_train.size(0))[:s]
    remaining_indices = torch.tensor([i for i in range(X_train.size(0)) if i not in sample_indices])
    return X_train[sample_indices],y_train[sample_indices],domain_y_train[sample_indices],\
           X_train[remaining_indices],y_train[remaining_indices],domain_y_train[remaining_indices]

def data_generator(train_features, train_labels,domain_train_labels, test_features, test_labels,domain_test_labels,batch_size):
    print(f'训练集维度{train_features.shape}，测试集维度{test_features.shape}\n'
          f'[样本数，1，样本长度]，875=125*7,7s')
    count_train = torch.bincount(train_labels)
    count_test = torch.bincount(test_labels)
    addc = torch.tensor([0])
    if len(count_test) == 2:
        count_test = torch.cat((count_test, addc), 0)
    print(f'训练集数据频度,VF0:{count_train[0]},SR1:{count_train[1]},MA2:{count_train[2]}')
    print(f'测试集数据频度,VF0:{count_test[0]},SR1:{count_test[1]},MA2:{count_test[2]}')
    train_len = len(train_labels)
    test_len = len(test_labels)
    train_iter = load_array((train_features, train_labels,domain_train_labels), batch_size, is_train=True)
    test_iter = load_array((test_features, test_labels,domain_test_labels), batch_size, is_train=False)
    return train_iter,test_iter,train_len,test_len

