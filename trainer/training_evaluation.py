"""
作者：yueyue
日期：2023年10月30日
"""
import math
import random

import numpy as np
import torch
from torch.nn import init
import scipy.io as scio
from torch import nn
from utils import accuracy,accuracy2
def domain_train(model,optimizer,criterion,data_loader,device):
    train_loss, train_acc = [], []
    # 建立模型
    model.train()
    for X, _,y in data_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        optimizer.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_acc.append(y.eq(out.detach().argmax(dim=1)).float().mean())

    total_loss = torch.tensor(train_loss).mean()
    total_acc = torch.tensor(train_acc).mean() * 100
    return total_loss, total_acc, torch.tensor(0),torch.tensor(0),torch.tensor(0)
def domain_test(label_classifier,test_iter,loss,device):
    label_classifier.eval()
    test_loss=[]
    test_acc=[]
    with torch.no_grad():
        for X, _,y in test_iter:
            X, y = X.to(device), y.to(device)
            out = label_classifier(X)
            l = loss(out, y)
            # 记录误差
            test_loss.append(l.item())
            # 记录准确率
            test_acc.append(y.eq(out.detach().argmax(dim=1)).float().mean())
    total_loss = torch.tensor(test_loss).mean()
    total_acc = torch.tensor(test_acc).mean() * 100
    return total_loss, total_acc, torch.tensor(0),torch.tensor(0),torch.tensor(0)
def train(model,optimizer,criterion,data_loader,device):
    train_loss, train_acc = [], []
    # 建立模型
    VF_acc, SR_acc, MA_acc = 0, 0, 0
    model.train()
    addc = torch.tensor([0]).to(device)
    count_train = torch.zeros(3).to(device)
    for X, y, _ in data_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        optimizer.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        ct_iter = torch.bincount(y.detach())
        if len(ct_iter) == 2:
            ct_iter = torch.cat((ct_iter, addc), 0)
        if len(ct_iter) == 1:
            ct_iter = torch.cat((ct_iter, addc), 0)
            ct_iter = torch.cat((ct_iter, addc), 0)
        count_train += ct_iter
        # print(count_train)
        corr_pred = accuracy(out.detach(), y.detach())
        train_acc.append(y.eq(out.detach().argmax(dim=1)).float().mean())
        VF_acc += corr_pred['VF']
        SR_acc += corr_pred['SR']
        MA_acc += corr_pred['MA']
    total_loss = torch.tensor(train_loss).mean()
    total_acc = torch.tensor(train_acc).mean() * 100
    return total_loss, total_acc, (VF_acc / count_train[0]).cpu(),(SR_acc / count_train[1]).cpu(), (MA_acc / count_train[2]).cpu()
def cross_grad_train(label_classifier,domain_classifier,optim_F,optim_D,loss,data_loader,device):
    train_loss, train_acc = [], []
    # 建立模型
    VF_acc, SR_acc, MA_acc = 0, 0, 0
    eps_f = 1.0
    eps_d = 1.0
    alpha_f = 0.5
    alpha_d = 0.5
    count_train = torch.zeros(3).to(device)
    label_classifier.train()
    domain_classifier.train()
    addc = torch.tensor([0]).to(device)
    for X, y, domain_y in data_loader:
        X, y, domain_y = X.to(device), y.to(device), domain_y.to(device)
        X.requires_grad = True
        # Compute domain perturbation
        out_D = domain_classifier(X)
        loss_d = loss(out_D, domain_y)
        loss_d.backward()
        grad_d = torch.clamp(X.grad.data, min=-0.5, max=0.5)
        input_d = X + eps_f * grad_d

        # Compute label perturbation
        X.grad.data.zero_()
        out_F = label_classifier(X)
        loss_f = loss(out_F, y)
        loss_f.backward()
        grad_f = torch.clamp(X.grad.data, min=-0.1, max=0.1)
        input_f = X + eps_d * grad_f
        X = X.detach()

        # update label net
        out_F_X = label_classifier(X)
        out_F_input_d = label_classifier(input_d)
        optim_F.zero_grad()
        loss_f1 = loss(out_F_X, y)
        loss_f2 = loss(out_F_input_d, y)
        loss_f = (1 - alpha_f) * loss_f1 + alpha_f * loss_f2
        loss_f.backward()
        optim_F.step()

        # Update domain net
        out_d_X = domain_classifier(X)
        out_d_input_f = domain_classifier(input_f)
        loss_d1 = loss(out_d_X, domain_y)
        loss_d2 = loss(out_d_input_f, domain_y)
        loss_d = (1 - alpha_d) * loss_d1 + alpha_d * loss_d2
        optim_D.zero_grad()
        loss_d.backward()
        optim_D.step()

        train_loss.append(loss_f.item())
        ct_iter = torch.bincount(y.detach())
        if len(ct_iter) == 2:
            ct_iter = torch.cat((ct_iter, addc), 0)
        if len(ct_iter) == 1:
            ct_iter = torch.cat((ct_iter, addc), 0)
            ct_iter = torch.cat((ct_iter, addc), 0)
        count_train += ct_iter
        # print(count_train)
        corr_pred = accuracy(out_F_X.detach(), y.detach())
        train_acc.append(y.eq(out_F_X.detach().argmax(dim=1)).float().mean())
        VF_acc += corr_pred['VF']
        SR_acc += corr_pred['SR']
        MA_acc += corr_pred['MA']
    total_loss = torch.tensor(train_loss).mean()
    total_acc = torch.tensor(train_acc).mean() * 100
    return total_loss, total_acc, (VF_acc / count_train[0]).cpu(),(SR_acc / count_train[1]).cpu(), MA_acc / count_train[2].cpu()


def model_evaluate(label_classifier,test_iter,loss,device):
    label_classifier.eval()
    VF_acc_test, SR_acc_test, MA_acc_test = 0, 0, 0
    test_loss=[]
    test_acc=[]
    count_test = torch.zeros(3).to(device)
    addc = torch.tensor([0]).to(device)
    with torch.no_grad():
        for X, y,_ in test_iter:
            X, y = X.to(device), y.to(device)
            out = label_classifier(X)
            l = loss(out, y)
            # 记录误差
            test_loss.append(l.item())
            # 记录准确率
            corr_pred = accuracy(out, y)
            test_acc.append(y.eq(out.detach().argmax(dim=1)).float().mean())
            VF_acc_test += corr_pred['VF']
            SR_acc_test += corr_pred['SR']
            MA_acc_test += corr_pred['MA']
            ct_iter = torch.bincount(y.detach())
            if len(ct_iter) == 2:
                ct_iter = torch.cat((ct_iter, addc), 0)
            if len(ct_iter) == 1:
                ct_iter = torch.cat((ct_iter, addc), 0)
                ct_iter = torch.cat((ct_iter, addc), 0)
            count_test += ct_iter
    total_loss = torch.tensor(test_loss).mean()
    total_acc = torch.tensor(test_acc).mean() * 100
    return total_loss, total_acc, (VF_acc_test / count_test[0]).cpu(),(SR_acc_test / count_test[1]).cpu(), (MA_acc_test / count_test[2]).cpu()


def model_evaluate2(label_classifier,test_iter,loss,device,test_len):
    label_classifier.eval()
    VF_acc_test, SR_acc_test, MA_acc_test = 0, 0, 0
    test_loss=[]
    test_acc= 0
    count_test = torch.zeros(3).to(device)
    addc = torch.tensor([0]).to(device)
    with torch.no_grad():
        for X, y,_ in test_iter:
            X, y = X.to(device), y.to(device)
            out = label_classifier(X)
            l = loss(out, y)
            # 记录误差
            test_loss.append(l.item())
            # 记录准确率
            acc,corr_pred = accuracy2(out, y)
            test_acc += acc
            VF_acc_test += corr_pred['VF']
            SR_acc_test += corr_pred['SR']
            MA_acc_test += corr_pred['MA']
            ct_iter = torch.bincount(y.detach())
            if len(ct_iter) == 2:
                ct_iter = torch.cat((ct_iter, addc), 0)
            if len(ct_iter) == 1:
                ct_iter = torch.cat((ct_iter, addc), 0)
                ct_iter = torch.cat((ct_iter, addc), 0)
            count_test += ct_iter
    total_loss = torch.tensor(test_loss).mean()
    total_acc = test_acc/test_len*100
    return total_loss, total_acc, (VF_acc_test / count_test[0]).cpu(),(SR_acc_test / count_test[1]).cpu(), (MA_acc_test / count_test[2]).cpu()


def model_evaluate3(label_classifier,test_iter,loss,device,test_len):
    label_classifier.eval()
    VF_acc_test, SR_acc_test, MA_acc_test = 0, 0, 0
    test_loss=[]
    test_acc= 0
    count_test = torch.zeros(3).to(device)
    addc = torch.tensor([0]).to(device)
    with torch.no_grad():
        for X, y,domain_y in test_iter:
            X, y = X.to(device), y.to(device)
            out,_ = label_classifier(X,0)
            l = loss(out, y)
            # 记录误差
            test_loss.append(l.item())
            # 记录准确率
            acc,corr_pred = accuracy2(out, y)
            test_acc += acc
            VF_acc_test += corr_pred['VF']
            SR_acc_test += corr_pred['SR']
            MA_acc_test += corr_pred['MA']
            ct_iter = torch.bincount(y.detach())
            if len(ct_iter) == 2:
                ct_iter = torch.cat((ct_iter, addc), 0)
            if len(ct_iter) == 1:
                ct_iter = torch.cat((ct_iter, addc), 0)
                ct_iter = torch.cat((ct_iter, addc), 0)
            count_test += ct_iter
    total_loss = torch.tensor(test_loss).mean()
    total_acc = test_acc/test_len*100
    return total_loss, total_acc, (VF_acc_test / count_test[0]).cpu(),(SR_acc_test / count_test[1]).cpu(), (MA_acc_test / count_test[2]).cpu()