import os
# import variational
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.ticker import FuncFormatter
# import os
# import time
# import math
# import pandas as pd
# from collections import OrderedDict
# from sklearn.linear_model import LogisticRegression
#
# import numpy as np
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import sys
sys.path.append("..")
from common import utils
from common import datasets
from common.resnet_1 import ResNet50
from common.resnet_1 import ResNet18
# import tqdm
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import itertools
from tqdm.autonotebook import tqdm
# from models import *
# import models
# from logger import *
from common.utils import *
from common.utils_1 import *
import argparse
import json
import os
import copy
import random
from collections import defaultdict

import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from io import BytesIO
import os
import errno
import argparse
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 30   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.1        #学习率
seed = 1
num_classes = 10
class_to_forget = None
num_to_forget = None
weight_decay = 0.9
# lossfn = 'mse'
lossfn = None
dataset = 'cifar10'


def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss


def run_train_epoch(model: nn.Module, model_init, data_loader: torch.utils.data.DataLoader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.SGD, split: str, epoch: int, ignore_index=None,
                    negative_gradient=False, negative_multiplier=-1, random_labels=False,
                    quiet=False, delta_w=None, scrub_act=False):
    model.eval()
    metrics = AverageMeter()
    # num_labels = data_loader.dataset.targets.max().item() + 1

    with torch.set_grad_enabled(split != 'test'):
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            input, target = batch
            output = model(input)
            if split == 'test' and scrub_act:
                G = []
                for cls in range(num_classes):
                    grads = torch.autograd.grad(output[0, cls], model.parameters(), retain_graph=True)
                    grads = torch.cat([g.view(-1) for g in grads])
                    G.append(grads)
                # grads = torch.autograd.grad(output_sf[0, cls], model_scrubf.parameters(), retain_graph=False)
                G = torch.stack(G).pow(2)
                delta_f = torch.matmul(G, delta_w)
                output += delta_f.sqrt() * torch.empty_like(delta_f).normal_()
            loss = loss_fn(output, target) + l2_penalty(model, model_init, weight_decay)
            metrics.update(n=input.size(0), loss=loss_fn(output, target).item(), error=get_error(output, target))

            if split != 'test':
                model.zero_grad()
                loss.backward()
                optimizer.step()
    # if not quiet:
    #     log_metrics(split, metrics, epoch)
    return metrics.avg


def test(model, data_loader):
    loss_fn = nn.CrossEntropyLoss()
    model_init=copy.deepcopy(model)
    return run_train_epoch(model, model_init, data_loader, loss_fn, optimizer=None, split='test', epoch=EPOCH, ignore_index=None, quiet=True)

# data loader
# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train) #训练数据集
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
print(len(trainset))
print(len(testset))

# 输入要删除的类别
forgetClasses = [8, 9]
forgottenExamples_train = []
unforgottenExamples_train = []
forgottenExamples_test = []
unforgottenExamples_test = []
for i, item in enumerate(trainset):
    # if i > 1000:
    #     break
    if item[1] in forgetClasses:
        forgottenExamples_train.append(item)
    else:
        unforgottenExamples_train.append(item)
for i, item in enumerate(testset):
    # if i > 1000:
    #     break
    if item[1] in forgetClasses:
        forgottenExamples_test.append(item)
    else:
        unforgottenExamples_test.append(item)

temp_test = []
for i, item in enumerate(testset):
    if i > 1000:
        break
    temp_test.append(item)


def replace_loader_dataset(data_loader, dataset, batch_size=BATCH_SIZE, seed=1, shuffle=True):
    torch.manual_seed(seed)
    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle)


forget_loader = torch.utils.data.DataLoader(forgottenExamples_train, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True)
retain_loader = torch.utils.data.DataLoader(unforgottenExamples_train, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True)
test_loader_full = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False)
# test_loader_full = torch.utils.data.DataLoader(temp_test, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=False)

# create models
root_dir = r'/home/ubuntu/ml/resnet18_cifar10/model/'
# root_dir = r'/media/public/ml/resnet18_cifar10/model/'
model_scrubf = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_normal_train_finished_saving_60.pth")
model_scrubf.load_state_dict(checkpoint)

modelf = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_normal_train_finished_saving_60.pth")
modelf.load_state_dict(checkpoint)

modelf0 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_forget_two_kinds_finished_saving_30_29_time_.pth")
modelf0.load_state_dict(checkpoint)

for p in itertools.chain(modelf.parameters(), modelf0.parameters(), model_scrubf.parameters()):
    p.data0 = copy.deepcopy(p.data.clone())


# %%
def hessian(dataset, model):
    model.eval()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    for p in model.parameters():
        p.grad_acc = 0
        p.grad2_acc = 0

    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad_acc += (orig_target == target).float() * p.grad.data
                    p.grad2_acc += prob[:, y] * p.grad.data.pow(2)
    for p in model.parameters():
        p.grad_acc /= len(train_loader)
        p.grad2_acc /= len(train_loader)


# %%
hessian(retain_loader.dataset, model_scrubf)
hessian(retain_loader.dataset, modelf)
hessian(retain_loader.dataset, modelf0)


# %%
def get_mean_var(p, is_base_dist=False, alpha=3e-6):
    var = copy.deepcopy(1. / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())
    if p.size(0) == num_classes and num_to_forget is None:
        mu[class_to_forget] = 0
        var[class_to_forget] = 0.0001
    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
    #         var*=1
    return mu, var


def kl_divergence_fisher(mu0, var0, mu1, var1):
    return ((mu1 - mu0).pow(2) / var0 + var1 / var0 - torch.log(var1 / var0) - 1).sum()


# %% md
## Fisher Noise in Weights
# %%
# Computes the amount of information not forgotten at all layers using the given alpha
# alpha = 1e-6
# total_kl = 0
# torch.manual_seed(seed)
# for (k, p), (k0, p0) in zip(modelf.named_parameters(), modelf0.named_parameters()):
#     mu0, var0 = get_mean_var(p, False, alpha=alpha)
#     mu1, var1 = get_mean_var(p0, True, alpha=alpha)
#     kl = kl_divergence_fisher(mu0, var0, mu1, var1).item()
#     total_kl += kl
#     print(k, f'{kl:.1f}')
# print("Total:", total_kl)
# %%
fisher_dir = []
alpha = 1e-6
torch.manual_seed(seed)
for i, p in enumerate(modelf.parameters()):
    mu, var = get_mean_var(p, False, alpha=alpha)
    p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
    fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())

# for i, p in enumerate(modelf0.parameters()):
#     mu, var = get_mean_var(p, False, alpha=alpha)
#     p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
# %%
print(test(modelf, retain_loader))
print(test(modelf, forget_loader))
print(test(modelf, test_loader_full))
print('Saving model......')
torch.save(modelf.state_dict(), root_dir+'resnet18_cifar10_fisher_forget_model_1.pth')
print('saved!')

def get_metrics(model,dataloader,criterion,samples_correctness=False,use_bn=False,delta_w=None,scrub_act=False):
    activations=[]
    predictions=[]
    if use_bn:
        model.train()
        dataloader = torch.utils.data.DataLoader(retain_loader.dataset, batch_size=128, shuffle=True)
        for i in range(10):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    model.eval()
    metrics = AverageMeter()
    mult = 0.5 if lossfn=='mse' else 1
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        loss = mult*criterion(output, target)
        if samples_correctness:
            activations.append(torch.nn.functional.softmax(output,dim=1).cpu().detach().numpy().squeeze())
            predictions.append(get_error(output,target))
        metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
    if samples_correctness:
        return metrics.avg,np.stack(activations),np.array(predictions)
    else:
        return metrics.avg


def activations_predictions(model,dataloader,name):
    criterion = torch.nn.CrossEntropyLoss()
    metrics,activations,predictions=get_metrics(model,dataloader,criterion,True)
    print(f"{name} -> Loss:{np.round(metrics['loss'],3)}, Error:{metrics['error']}")
    return activations,predictions


def predictions_distance(l1,l2,name):
    dist = np.sum(np.abs(l1-l2))
    print(f"Predictions Distance {name} -> {dist}")


def activations_distance(a1,a2,name):
    dist = np.linalg.norm(a1-a2,ord=1,axis=1).mean()
    print(f"Activations Distance {name} -> {dist}")


m0_D_r_activations, m0_D_r_predictions = activations_predictions(modelf0, retain_loader, 'Retrain_Model_D_r')
m0_D_f_activations, m0_D_f_predictions = activations_predictions(modelf0, forget_loader, 'Retrain_Model_D_f')
m0_D_t_activations, m0_D_t_predictions = activations_predictions(modelf0, test_loader_full, 'Retrain_Model_D_t')
# %%
fisher_D_r_activations, fisher_D_r_predictions = activations_predictions(modelf, retain_loader, 'Fisher_D_r')
fisher_D_f_activations, fisher_D_f_predictions = activations_predictions(modelf, forget_loader, 'Fisher_D_f')
fisher_D_t_activations, fisher_D_t_predictions = activations_predictions(modelf, test_loader_full, 'Fisher_D_t')
# %%
predictions_distance(m0_D_f_predictions, fisher_D_f_predictions, 'Retrain_Fisher_D_f')
activations_distance(m0_D_f_activations, fisher_D_f_activations, 'Retrain_Fisher_D_f')
activations_distance(m0_D_r_activations, fisher_D_r_activations, 'Retrain_Fisher_D_r')
activations_distance(m0_D_t_activations, fisher_D_t_activations, 'Retrain_Fisher_D_t')
