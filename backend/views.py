from django.shortcuts import render

# Create your views here.
import json
from django.http import HttpResponse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from common import utils
from common import datasets
from common.resnet_1 import ResNet50
from common.resnet_1 import ResNet18
import os
import numpy as np
import datetime


def hello(request):
    resp = {'message': "success", 'result': 'ok'}
    resp['message'] = 'aaddDa'
    resp['result'] = 'ok'
    resp['data'] = "aaa"
    return HttpResponse(json.dumps(resp), content_type="application/json")

def predict(request):

    params = request.GET;
    print(params)
    print(params['params'].split("_"))
    # return HttpResponse(json.dumps([]), content_type="application/json")
    fileList = params['params'].split("_")
    start = datetime.datetime.now()
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 超参数设置
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)
    LR = 0.1  # 学习率
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    SAMPLE_SIZE = 10

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    forgetClasses = [8, 9]
    forgottenExamples = []
    unforgottenExamples = []
    testExamples = []
    for i, item in enumerate(testset):
        if i > SAMPLE_SIZE:
            break
        testExamples.append(item)
        if item[1] in forgetClasses:
            forgottenExamples.append(item)
        else:
            unforgottenExamples.append(item)
    # Cifar-10的标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    print("Waiting Test!")

    targetFile = 'resnet18_cifar10_forget_two_kinds_20210321_25_machine_1.pth'
    savedFiles = [
        # 'resnet18_cifar10_normal_train_finished_saving_60.pth',
        'resnet18_cifar10_forget_two_kinds_finished_saving_20210515_21.pth',
        'resnet18_cifar10_train_to_forget_two_kinds_finished_saving_20210515_11.pth',
        # 'resnet18_cifar10_noraml_train_init.pth',
        # 'resnet18_cifar10_fc_before_training.pth',
        # 'resnet18_cifar10_fc_before_training.pth_forget_two_kinds_after_finetuning_30.pth',
        # 'resnet18_cifar10_fc_conv17_before_training.pth',
        # 'resnet18_cifar10_forget_two_kinds_init.pth',
    ]
    testloader_unforget = torch.utils.data.DataLoader(unforgottenExamples, batch_size=100, shuffle=False, num_workers=2)
    testloader_forget = torch.utils.data.DataLoader(forgottenExamples, batch_size=100, shuffle=False, num_workers=2)
    testloader_all = torch.utils.data.DataLoader(testExamples, batch_size=100, shuffle=False, num_workers=2)
    # testloader_all = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    # 测试准确率
    totals = []
    corrects = []
    for i in range(len(savedFiles)):
        totals.append(0)
        corrects.append(0)

    returnData = []
    path = r'D:\ww2\defense-backend\model\\'

    queryFiles = []
    for i in fileList:
        queryFiles.append(savedFiles[int(i)])


    with torch.no_grad():
        for i, file in enumerate(queryFiles, 0):
            # net.load_state_dict("./model/" + file, map_location='cpu')
            checkpoint = torch.load(path + file)
            net.load_state_dict(checkpoint)
            map = {}
            for inner_i in range(0, 10):
                if inner_i not in map:
                    map[inner_i] = {}
                for inner_j in range(0, 10):
                    map[inner_i][inner_j] = 0
            for data in testloader_unforget:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                predictedList = predicted.cpu().numpy().tolist()
                labelsList = labels.cpu().numpy().tolist()
                for itemCnt, item in enumerate(predictedList):
                    map[labelsList[itemCnt]][item] += 1
                totals[i] += labels.size(0)
                corrects[i] += (predicted == labels).sum()

            dataItem = []
            for key1 in map:
                for key2 in map[key1]:
                    # item = {'label': key1, 'predict': key2, 'value': map[key1][key2]}
                    item = {'ground_truth': key1, 'predict': key2, 'count': map[key1][key2]}
                    dataItem.append(item)
            returnData.append(dataItem)

        for i, file in enumerate(queryFiles, 0):
            print(file + '测试保留集分类准确率为：%.3f%%' % (100. * corrects[i] / totals[i]))

    end = datetime.datetime.now()
    resp = {}
    resp['message'] = str(end - start)
    resp['result'] = 'ok'
    resp['data'] = returnData
    print(end - start)
    return HttpResponse(json.dumps(resp), content_type="application/json")