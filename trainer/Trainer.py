"""
作者：yueyue
日期：2023年10月30日
"""
import torch
import torch.nn as nn
from trainer.training_evaluation import cross_grad_train,model_evaluate,train
from args import args
from config_files.configs import Config as Configs
from utils import adjust_lr
from utils import count_parameters
from dataloader.dataloader import data_generator,get_K_fold_crossvalidation_data,Bcgdataset
import random
config = Configs()

def normal_Trainer(model,train_dl,test_dl,device):
    train_ls, test_ls = [], []
    train_acces, test_acces = [], []
    train_VF_acc_set, train_SR_acc_set, train_MA_acc_set = [], [], []
    valid_VF_acc_set, valid_SR_acc_set, valid_MA_acc_set = [], [], []
    label_classifier = model(in_channels=1, channels=64, num_classes=config.label_num_classes, block_num=None).to(
        device)
    print(f'The model has {count_parameters(label_classifier):,} trainable parameters')
    optimizer = torch.optim.Adam(label_classifier.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
                               weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.num_epoch):
        adjust_lr(config.lr, epoch, optimizer)
        train_loss, train_acc, VF_acc, SR_acc, MA_acc = train(label_classifier,optimizer,criterion,train_dl,device)
        test_loss, test_acc, VF_test_acc, SR_test_acc, MA_test_acc = model_evaluate(label_classifier, test_dl,
                                                                                    criterion, device)
        if config.save_ckp and epoch % 10 == 0:
            print(f'输出当前{epoch}epoch的训练')
            log_dirfile_best = 'ck_p2/checkpoint_epoch=' + str(epoch) + 'normaltrain.pt'
            torch.save(label_classifier.state_dict(), log_dirfile_best)
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f},  VF acc: {:.4f},SR acc: {:.4f},MA acc:{:.4f}\n'
              '          Test Loss: {:.4f}, Test Acc: {:.4f},  VF acc:{:.4f},SR acc:{:.4f},MA acc:{:.4f}'
              .format(epoch, train_loss, train_acc, VF_acc, SR_acc, MA_acc,
                      test_loss, test_acc, VF_test_acc, SR_test_acc, MA_test_acc))
    train_ls.append(train_loss)
    test_ls.append(test_loss)
    train_acces.append(train_acc)
    test_acces.append(test_acc)
    train_VF_acc_set.append(VF_acc)
    train_SR_acc_set.append(SR_acc)
    train_MA_acc_set.append(MA_acc)
    valid_VF_acc_set.append(VF_test_acc)
    valid_SR_acc_set.append(SR_test_acc)
    valid_MA_acc_set.append(MA_test_acc)
    return train_ls, test_ls, train_acces, test_acces, train_VF_acc_set, train_SR_acc_set, train_MA_acc_set, valid_VF_acc_set, valid_SR_acc_set, valid_MA_acc_set
def cross_grad_Trainer(model,train_dl,test_dl,device):
    train_ls, test_ls = [], []
    train_acces, test_acces = [], []
    train_VF_acc_set, train_SR_acc_set, train_MA_acc_set = [], [], []
    valid_VF_acc_set, valid_SR_acc_set, valid_MA_acc_set = [], [], []
    label_classifier = model(in_channels=1, channels=64, num_classes=config.label_num_classes, block_num=None).to(device)
    domain_classifier = model(in_channels=1, channels=64, num_classes=config.domain_num_classes, block_num=None).to(device)
    print(f'The model has {count_parameters(label_classifier):,} trainable parameters')
    optim_F = torch.optim.Adam(label_classifier.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
                               weight_decay=config.weight_decay)
    optim_D = torch.optim.Adam(domain_classifier.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
                               weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.num_epoch):
        adjust_lr(config.lr,epoch, optim_D)
        adjust_lr(config.lr,epoch, optim_F)
        train_loss, train_acc,VF_acc,SR_acc,MA_acc = cross_grad_train(label_classifier,domain_classifier,optim_F,optim_D,criterion,train_dl,device)
        test_loss, test_acc, VF_test_acc, SR_test_acc, MA_test_acc = model_evaluate(label_classifier,test_dl,criterion,device)
        if config.save_ckp and epoch % 10 == 0:
            print(f'输出当前{epoch}epoch的训练')
            log_dirfile_best = 'ck_p2/checkpoint_epoch=' + str(epoch) + 'crossgrad.pt'
            torch.save(label_classifier.state_dict(), log_dirfile_best)
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f},  VF acc: {:.4f},SR acc: {:.4f},MA acc:{:.4f}\n'
              '          Test Loss: {:.4f}, Test Acc: {:.4f},  VF acc:{:.4f},SR acc:{:.4f},MA acc:{:.4f}'
              .format(epoch, train_loss, train_acc,VF_acc,SR_acc,MA_acc,
                      test_loss, test_acc, VF_test_acc, SR_test_acc, MA_test_acc))
    train_ls.append(train_loss)
    test_ls.append(test_loss)
    train_acces.append(train_acc)
    test_acces.append(test_acc)
    train_VF_acc_set.append(VF_acc)
    train_SR_acc_set.append(SR_acc)
    train_MA_acc_set.append(MA_acc)
    valid_VF_acc_set.append(VF_test_acc)
    valid_SR_acc_set.append(SR_test_acc)
    valid_MA_acc_set.append(MA_test_acc)
    return train_ls, test_ls, train_acces, test_acces, train_VF_acc_set, train_SR_acc_set, train_MA_acc_set, valid_VF_acc_set, valid_SR_acc_set, valid_MA_acc_set

def k_fold(model, k, dataset, device, pig_train_set):
    '''K折交叉验证训练'''
    train_l_sum, valid_l_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    t_VF, t_SR, t_MA = 0, 0, 0
    v_VF, v_SR, v_MA = 0, 0, 0
    random.shuffle(pig_train_set)
    print('打乱后的训练集', pig_train_set)
    for i in range(k):
        data = get_K_fold_crossvalidation_data(dataset=dataset, pigset=pig_train_set, K=config.k_fold_num, i=i)
        print(f'第{i}折训练开始')
        train_iter,test_iter,train_len,test_len = data_generator(*data,config.batch_size)
        train_ls, valid_ls, train_acc, valid_acc, train_VF_acc_set, train_SR_acc_set, train_MA_acc_set, valid_VF_acc_set, valid_SR_acc_set, valid_MA_acc_set \
            = cross_grad_Trainer(model,train_iter,test_iter,device)
        valid_l_sum += valid_ls[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += valid_acc[-1]
        t_VF += train_VF_acc_set[-1]
        t_SR += train_SR_acc_set[-1]
        t_MA += train_MA_acc_set[-1]
        v_VF += valid_VF_acc_set[-1]
        v_SR += valid_SR_acc_set[-1]
        v_MA += valid_MA_acc_set[-1]
        print(f'fold{i + 1},trainloss{float(train_ls[-1]):f},validloss{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k, train_acc_sum / k, valid_acc_sum / k, t_VF / k, t_SR / k, t_MA / k, v_VF / k, v_SR / k, v_MA / k

def k_fold_normal(model, k, dataset, device, pig_train_set):
    '''K折交叉验证训练'''
    train_l_sum, valid_l_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    t_VF, t_SR, t_MA = 0, 0, 0
    v_VF, v_SR, v_MA = 0, 0, 0
    random.shuffle(pig_train_set)
    print('打乱后的训练集', pig_train_set)
    for i in range(k):
        data = get_K_fold_crossvalidation_data(dataset=dataset, pigset=pig_train_set, K=config.k_fold_num, i=i)
        print(f'第{i}折训练开始')
        train_iter,test_iter,train_len,test_len = data_generator(*data,config.batch_size)
        train_ls, valid_ls, train_acc, valid_acc, train_VF_acc_set, train_SR_acc_set, train_MA_acc_set, valid_VF_acc_set, valid_SR_acc_set, valid_MA_acc_set \
            = normal_Trainer(model,train_iter,test_iter,device)
        valid_l_sum += valid_ls[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += valid_acc[-1]
        t_VF += train_VF_acc_set[-1]
        t_SR += train_SR_acc_set[-1]
        t_MA += train_MA_acc_set[-1]
        v_VF += valid_VF_acc_set[-1]
        v_SR += valid_SR_acc_set[-1]
        v_MA += valid_MA_acc_set[-1]
        print(f'fold{i + 1},trainloss{float(train_ls[-1]):f},validloss{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k, train_acc_sum / k, valid_acc_sum / k, t_VF / k, t_SR / k, t_MA / k, v_VF / k, v_SR / k, v_MA / k
