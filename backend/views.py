from django.shortcuts import render

# Create your views here.
import json
from django.http import HttpResponse
import sys
sys.path.append("..")
from common.resnet_1 import ResNet18
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from common.utils import generateReverseParamsResnet18
from common.resnet_for_mnist import ResNet18 as MNISTResNet18
from common.resnet_1 import ResNet18 as CIFARResNet18
import os
import time
from common.lr_scheduler_temp import ReduceLROnPlateau
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import math

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

def hello(request):
    resp = {'message': "success", 'result': 'ok'}
    resp['message'] = 'aaddDa'
    resp['result'] = 'ok'
    resp['data'] = "aaa"
    return HttpResponse(json.dumps(resp), content_type="application/json")


def getModelResult(request):
    print(request.body)
    params = json.loads(request.body)
    structure = params['selectedStructure']
    selectedLRFModel = params['selectedLRFModel']
    selectedCompareModel = params['selectedCompareModel']
    pictures = params['pictures']
    modelName, dataset = structure.split('+')
    print(structure)
    print(selectedLRFModel)
    print(selectedCompareModel)
    splitDatasetByKind = {}
    train_ds, test_ds = getDataset(dataset)
    for i, item in enumerate(test_ds):
        data, label = item
        if splitDatasetByKind.get(label) is None:
            splitDatasetByKind[label] = [item]
        else:
            splitDatasetByKind[label].append(item)
    resp = {'message': "success", 'result': 'ok', 'data': "aaa"}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def getPictureResult(request):
    print(request.body)
    params = json.loads(request.body)
    structure = params['selectedStructure']
    selectedLRFModel = params['selectedLRFModel']
    selectedCompareModel = params['selectedCompareModel']
    pictures = params['pictures']
    modelName, dataset = structure.split('+')
    print(structure)
    print(selectedLRFModel)
    print(selectedCompareModel)
    train_ds, test_ds = getDataset(dataset)

    # b = [i[1] for i in testList]
    # print(b)

    modelList = [selectedLRFModel, selectedCompareModel]
    fileRoot = getFilerootName()
    filePath = fileRoot + "/model/"
    results = []
    item = {
        "name": "",
        "lrf_predict": "",
        "compare_predict": "",
        "lrf_vector": [],
        "compare_vector": [],
    }
    # 准备测试数据
    testList = []
    for i in range(len(pictures)):
        index, _ = pictures[i].split('.')
        testList.append(test_ds[int(index)])

    dataLoader = DataLoader(testList, batch_size=100, shuffle=False)
    print(len(dataLoader.dataset))

    net = getNet(dataset)

    lrfPredicted = []
    lrfOutput = []
    comparePredicted = []
    compareOutput = []
    sm = nn.Softmax()
    for i, data in enumerate(dataLoader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 加载LRF模型
        checkpoint = torch.load(filePath + selectedLRFModel, map_location='cpu')
        net.load_state_dict(checkpoint)
        net.eval()
        lrfOutput = net(images)
        # 加载对照模型
        checkpoint = torch.load(filePath + selectedCompareModel, map_location='cpu')
        net.load_state_dict(checkpoint)
        net.eval()
        compareOutput = net(images)
        break
    with torch.no_grad():
        for i, picItem in enumerate(pictures):
            _, lrfPredicted = torch.max(lrfOutput.data, 1)
            lrfVectorList = sm(lrfOutput[i].cpu().view(10)).numpy().tolist()
            # lrfVectorList = lrfOutput[i].cpu().numpy().tolist()

            # lrfVectorList = [round(j, 4) for j in lrfVectorList]
            _, comparePredicted = torch.max(compareOutput.data, 1)
            cmpVectorList = sm(compareOutput[i].cpu().view(10)).numpy().tolist()
            # cmpVectorList = compareOutput[i].cpu().numpy().tolist()
            # cmpVectorList = [round(j, 4) for j in cmpVectorList]
            item = {}
            item['name'] = picItem
            item['lrf_predict'] = lrfPredicted[i].item()
            item['lrf_vector'] = lrfVectorList
            item['compare_predict'] = comparePredicted[i].item()
            item['compare_vector'] = cmpVectorList
            results.append(item)
            print(item)
    resp = {'message': "success", 'result': 'ok', 'data': results}
    return HttpResponse(json.dumps(resp), content_type="application/json")

def uploadFile(request):
    print(request.FILES['file'])
    fileName = str(request.FILES['file'])
    index, _ = fileName.split('.')
    print(index)
    print(int(index))
    resp = {'message': "success", 'result': 'ok'}
    resp['message'] = 'aaddDa'
    resp['result'] = 'ok'
    resp['data'] = "aaa"
    return HttpResponse(json.dumps(resp), content_type="application/json")

def getModelList(request):
    filePath = r'D:\www\defense-backend\backend\model'
    fileList = os.listdir(filePath)
    print(fileList)
    resp = {'message': "success", 'result': 'ok', 'data': fileList}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def generateParamsResnet18(net, former, later, layeredParams, filePath, strucName, datasetName, layer):
    # 模型定义-ResNet
    # net = ResNet18().to(device)
    checkpoint = torch.load(filePath+former, map_location='cpu')
    net.load_state_dict(checkpoint)
    params = net.state_dict()
    toLoad = params
    checkpoint = torch.load(filePath+later, map_location='cpu')
    fileNameList = []
    freezeParamsList = []
    for i, params in enumerate(layeredParams):
        if i + 1 in layer:
            newLayerParams = []
            for j in range(i+1):
                newLayerParams = newLayerParams + layeredParams[len(layeredParams)-j-1]
            freezeParams = []

            for j in range(len(layeredParams) - i - 1):
                freezeParams = freezeParams + layeredParams[j]
            resetLayerName = "reset_" + str(i+1) + "_"
            fileName = strucName+datasetName+resetLayerName+"before_training.pth"
            for k in checkpoint.keys():
                if k in newLayerParams:
                    toLoad[k] = checkpoint[k]
                    # print("added:" + k)
            net.load_state_dict(toLoad)
            print('Saving model:'+fileName)
            torch.save(net.state_dict(), '%s/%s' % (filePath, fileName))
            fileNameList.append(fileName)
            freezeParamsList.append(freezeParams)
    return fileNameList, freezeParamsList


def resetparam(request):
    print("get point start")
    # 解析请求
    print(request.body)
    body = request.body
    params = json.loads(body)
    structure = params['selectedStructure']
    initModel = params['selectedInitModel']
    normalModel = params['selectedNormalModel']
    resetNumber = params['resetNumber']
    modelName, dataset = structure.split('+')
    print(modelName)
    print(dataset)
    print(initModel)
    print(normalModel)
    print(resetNumber)
    net = getNet(dataset)
    layeredParams = getLayeredParams()
    fileRoot = getFilerootName()
    filePath = fileRoot + "/model/"
    fileNameList, freezeParamsList = generateParamsResnet18(net, normalModel, initModel, layeredParams, filePath, modelName+"_", dataset+"_", [resetNumber])
    resp = {'message': "success", 'result': 'ok', 'data': fileNameList[0]}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def forgettrain(request):
    start = time.time()
    print("forget train start")
    # 解析请求
    print(request.body)
    body = request.body
    params = json.loads(body)
    structure = params['selectedStructure']
    resetModel = params['selectedResetModel']
    forgetKinds = params['forgetKinds']
    modelName, dataset = structure.split('+')
    forgetList = forgetKinds.split('-')
    forgetList = list(map(int, forgetList))
    print(modelName)
    print(dataset)
    print(resetModel)
    print(forgetList)
    fileRoot = getFilerootName()
    # 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    filePath = fileRoot + "/model/"
    logPath = fileRoot + "/logs/"
    strucName = 'resnet18_'
    datasetName = dataset + '_'
    methodName = "forget_" + forgetKinds + "_kind_"
    logName = logPath + strucName + datasetName + methodName + "_test2_"
    pthName = filePath + strucName + datasetName + methodName + "_test2_"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 超参数设置
    if dataset == 'MNIST':
        EPOCH = 10  # 遍历数据集次数
        LR = 0.005  # 学习率
    else:
        EPOCH = 30
        LR = 0.1  # 学习率
    pre_epoch = 0  # 定义已经遍历数据集的次数
    BATCH_SIZE = 32  # 批处理尺寸(batch_size)
    accThreshold = 0.01
    criterion = nn.CrossEntropyLoss()  # 定义损失函数和优化方式，损失函数为交叉熵，多用于多分类问题

    # 获取数据集
    train_ds, test_ds = getDataset(dataset)
    # 获取保留训练集、保留测试集、遗忘训练集、遗忘测试集
    trainRetain, trainForget, testRetain, testForget = getSplitDataset(train_ds, test_ds, forgetList)
    trainloader, testloader = getDataloader(trainRetain, testRetain, BATCH_SIZE)
    forgetTestLoader = DataLoader(testForget, batch_size=BATCH_SIZE, shuffle=True)
    print('after loader')
    # 模型定义-ResNet
    net = getNet(dataset)
    optimizer = getOptimizer(net, LR)
    print('before train')
    model, _, forgetTestAcc = train(net, trainloader, testloader, forgetTestLoader, [], pre_epoch,
                                    EPOCH, resetModel, device, optimizer, criterion, filePath, logName, pthName)
    if forgetTestAcc > 1:
        forgetTestAcc /= 100
    end = time.time()
    span = end - start
    resp = {'message': "forgettrain", 'result': 'ok', 'data': model + "+++" + str(span)}
    return HttpResponse(json.dumps(resp), content_type="application/json")


# 获取文件目录
def getFilerootName():
    # 2080机器
    # fileRoot = r'/home/ubuntu/ml/resnet18-vggface100-2'
    # dataRoot = r'/home/ubuntu/ml/resnet18_vggface2'
    # datasetRoot = r'/datasets/train'

    # 1080机器
    # fileRoot = r'/media/public/ml/resnet18-vggface100-2'
    # dataRoot = r'/media/public/ml/resnet18_vggface2'
    # datasetRoot = r'/datasets/data/root'

    # 实验室台式机
    # fileRoot = r'D:\ww2\graduate_expriment\resnet18-vggface100-2'
    # dataRoot = r'D:\ww2\graduate_expriment\resnet18_vggface2'
    # datasetRoot = r'\datasets\data\root'
    # trainForgetFile = r"\train-20kinds-all.txt"
    # trainRetainFile = r"\train-80kinds-all.txt"
    # testForgetFile = r"\test-20kinds-all.txt"
    # testRetainFile = r"\test-80kinds-all.txt"
    # trainFile = r"\train_list_100.txt"
    # testFile = r"\test_list_100.txt"

    # 自己电脑
    fileRoot = r"D:\www\defense-backend\backend"
    # fileRoot = r'D:\www\graduate_expriment\resnet18-mnist'
    # dataRoot = r'D:\www\graduate_expriment\resnet18_vggface2'
    # datasetRoot = r'\datasets\data\root'
    return fileRoot


def getLayeredParams():
    layeredParams = []

    layeredParams.append(["conv1.0.weight", "conv1.1.weight", "conv1.1.bias"])
    layeredParams.append(["layer1.0.left.0.weight", "layer1.0.left.1.weight", "layer1.0.left.1.bias", ])
    layeredParams.append(["layer1.0.left.3.weight", "layer1.0.left.4.weight", "layer1.0.left.4.bias", ])
    layeredParams.append(["layer1.1.left.0.weight", "layer1.1.left.1.weight", "layer1.1.left.1.bias", ])
    layeredParams.append(["layer1.1.left.3.weight", "layer1.1.left.4.weight", "layer1.1.left.4.bias", ])

    layeredParams.append(["layer2.0.left.0.weight", "layer2.0.left.1.weight", "layer2.0.left.1.bias", ])
    layeredParams.append(
        ["layer2.0.left.3.weight", "layer2.0.left.4.weight", "layer2.0.left.4.bias", "layer2.0.shortcut.0.weight",
         "layer2.0.shortcut.1.weight", "layer2.0.shortcut.1.bias", ])
    layeredParams.append(["layer2.1.left.0.weight", "layer2.1.left.1.weight", "layer2.1.left.1.bias", ])
    layeredParams.append(["layer2.1.left.3.weight", "layer2.1.left.4.weight", "layer2.1.left.4.bias", ])

    layeredParams.append(["layer3.0.left.0.weight", "layer3.0.left.1.weight", "layer3.0.left.1.bias", ])
    layeredParams.append(
        ["layer3.0.left.3.weight", "layer3.0.left.4.weight", "layer3.0.left.4.bias", "layer3.0.shortcut.0.weight",
         "layer3.0.shortcut.1.weight", "layer3.0.shortcut.1.bias", ])
    layeredParams.append(["layer3.1.left.0.weight", "layer3.1.left.1.weight", "layer3.1.left.1.bias", ])
    layeredParams.append(["layer3.1.left.3.weight", "layer3.1.left.4.weight", "layer3.1.left.4.bias"])

    layeredParams.append(["layer4.0.left.0.weight", "layer4.0.left.1.weight", "layer4.0.left.1.bias", ])
    layeredParams.append(
        ["layer4.0.left.3.weight", "layer4.0.left.4.weight", "layer4.0.left.4.bias", "layer4.0.shortcut.0.weight",
         "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias", ])
    layeredParams.append(["layer4.1.left.0.weight", "layer4.1.left.1.weight", "layer4.1.left.1.bias", ])
    layeredParams.append(["layer4.1.left.3.weight", "layer4.1.left.4.weight", "layer4.1.left.4.bias", ])

    layeredParams.append(["fc.weight", "fc.bias", ])
    return layeredParams


def getDataset(dataset):
    if dataset == 'MNIST':
        train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
        test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), (0.5))])
        train_ds = CIFAR10('cifar10', download=True, train=True, transform=transform)
        test_ds = CIFAR10('cifar10', download=True, train=False, transform=transform)
    print(len(train_ds))
    print(len(test_ds))
    return train_ds, test_ds


def getDataloader(trainset, testset, batch_size):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader


def getSplitDataset(trainset, testset, forgetList):
    trainRetain = []
    testRetain = []
    trainForget = []
    testForget = []
    for item in trainset:
        data, label = item
        if label not in forgetList:
            trainRetain.append(item)
        else:
            trainForget.append(item)
    for item in testset:
        data, label = item
        if label not in forgetList:
            testRetain.append(item)
        else:
            testForget.append(item)
    print("train retain: "+str(len(trainRetain)))
    print("test retain: "+str(len(testRetain)))
    return trainRetain, trainForget, testRetain, testForget


def getOptimizer(net, LR):
    # return optim.SGD(net.parameters(), lr=LR, momentum=0.9,
    #                       weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    return torch.optim.RMSprop(net.parameters(), lr=LR)


def testAcc(net, testloader, criterion, device):
    correct = 0.0
    total = 0.0
    sum_loss = 0
    for iTest, data in enumerate(testloader):
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / total, sum_loss / len(testloader)


def getNet(dataset):
    if dataset == 'MNIST':
        net = MNISTResNet18().to(device)
    else:
        net = CIFARResNet18().to(device)
    return net

def train(net, trainloader, testloader, forgetTestLoader, frozenList, pre_epoch, EPOCH, preModel, device, optimizer, criterion, filePath, logName, pthName):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                  eps=1e-08)
    if preModel != "":
        checkpoint = torch.load(filePath + preModel)
        net.load_state_dict(checkpoint)
    fileAccName = logName + "_acc.txt"
    fileLogName = logName + "_log.txt"

    # 冻结相关层
    frozenIndex = []
    paramCount = 0
    for name, paramItem in net.named_parameters():
        if name in frozenList:
            frozenIndex.append(paramCount)
        paramCount = paramCount + 1
    fIndex = 0
    for paramItem in net.parameters():
        paramItem.requires_grad = True
        if fIndex in frozenIndex:
            paramItem.requires_grad = False  # 冻结网络
        fIndex = fIndex + 1
    # 训练
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    best_acc = 0
    tolerate = 2
    saveModelSpan = 10
    T_threshold = 0.0111
    bestModelName = None
    with open(fileAccName, "a+")as f:
        with open(fileLogName, "a+")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # if i > len(trainloader) * 0.01:
                    #     break
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    lastTrainLoss = sum_loss / (i + 1)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s | File: %s | LR: %.6f'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), preModel,
                             optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.6f'
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('\n')
                    f2.flush()
                f.write("last train loss: %.4f\n" % lastTrainLoss)
                f.flush()
                # 保留集测试和遗忘集测试准确率
                print("Waiting Test!")
                with torch.no_grad():
                    testRetainAcc, testLoss = testAcc(net, testloader, criterion, device)
                    testForgetAcc, testForgetLoss = testAcc(net, forgetTestLoader, criterion, device)
                    print('测试分类准确率为：%.3f%%, 当前学习率： %.3f, last loss: %.3f' % (
                        100. * testRetainAcc, optimizer.state_dict()['param_groups'][0]['lr'], testLoss))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f,train BATCH_SIZE:%d,lastLoss:%.3f\n" % (
                        epoch + 1, 100. * testRetainAcc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        optimizer.state_dict()['param_groups'][0]['lr'], trainloader.batch_size, testLoss))
                    print('遗忘集测试分类准确率为：%.3f%%' % (100. * testForgetAcc))
                    f.write('遗忘集测试分类准确率为：%.3f%%\n' % (100. * testForgetAcc))
                    f.flush()
                scheduler.step(testLoss, epoch=epoch)
                # 保存模型
                if testRetainAcc > best_acc:
                    best_acc = testRetainAcc
                    print('Saving best acc model......')
                    bestModelName = pthName+"_best_acc_model.pth"
                    torch.save(net.state_dict(), bestModelName)
                    f.write("save best model\n")
                    f.flush()
                if (epoch + 1) % saveModelSpan < 1:
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s_%03d_epoch.pth' % (pthName + preModel.replace("before", "after"), epoch + 1))
                # 检查loss是否达到阈值，如果达到阈值则停止训练
                if lastTrainLoss < T_threshold and epoch > tolerate:
                    print('train loss达到限值%s，提前退出' % lastTrainLoss)
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s_%03d_epoch.pth' % (pthName + preModel.replace("before", "after"), epoch + 1))
                    f.write("train loss %s 达到限值，提前退出\n" % lastTrainLoss)
                    f.flush()
                    break
                # 检查学习率lr是否达到阈值，如果达到阈值则停止训练
                if optimizer.state_dict()['param_groups'][0]['lr'] < 0.003:
                    print("学习率过小，退出")
                    f.write("学习率过小，退出\n")
                    f.flush()
                    break

            print('Saving model......')
            torch.save(net.state_dict(), '%s_%03d_epoch.pth' % (pthName, epoch + 1))
            print("Training Finished, TotalEPOCH=%03d" % (epoch+1))
            return bestModelName, best_acc, testForgetAcc

def getpoint(request):
# def getpoint():
    print("get point start")
    # 解析请求
    print(request.body)
    body = request.body
    params = json.loads(body)
    structure = params['selectedStructure']
    initModel = params['selectedInitModel']
    normalModel = params['selectedNormalModel']
    modelName, dataset = structure.split('+')
    print(modelName)
    print(dataset)
    print(initModel)
    print(normalModel)

    fileRoot = getFilerootName()
    layeredParams = getLayeredParams()

    # 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    filePath = fileRoot + "/model/"
    logPath = fileRoot + "/logs/"
    if dataset == 'MNIST':
        initModel = "resnet18_mnist_noraml_train_init.pth"
        finishedModel = "resnet18_mnist_normal_train_20.pth"
    else:
        initModel = "resnet18_cifar10_noraml_train_init.pth"
        finishedModel = "resnet18_cifar10_normal_train_finished_saving_60.pth"
    strucName = 'resnet18_'
    datasetName = dataset+'_'
    methodName = "reverse_forget_one_kind_reset_"
    logName = logPath + strucName + datasetName + methodName + "_test1_"
    pthName = filePath + strucName + datasetName + methodName + "_test1_"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 超参数设置
    if dataset == 'MNIST':
        EPOCH = 10  # 遍历数据集次数
        LR = 0.005  # 学习率
    else:
        EPOCH = 30
        LR = 0.1  # 学习率
    pre_epoch = 0  # 定义已经遍历数据集的次数
    BATCH_SIZE = 32  # 批处理尺寸(batch_size)
    accThreshold = 0.01
    criterion = nn.CrossEntropyLoss() # 定义损失函数和优化方式，损失函数为交叉熵，多用于多分类问题
    forget = [3]

    # 获取数据集
    train_ds, test_ds = getDataset(dataset)
    # 获取保留训练集、保留测试集、遗忘训练集、遗忘测试集
    trainRetain, trainForget, testRetain, testForget = getSplitDataset(train_ds, test_ds, forget)
    trainloader, testloader = getDataloader(trainRetain, testRetain, BATCH_SIZE)
    forgetTestLoader = DataLoader(testForget, batch_size=BATCH_SIZE, shuffle=True)

    # 模型定义-ResNet
    net = getNet(dataset)
    # 二分寻找分层点
    head = 1
    tail = 18
    while head < tail-1:
        mid = math.ceil((head + tail) / 2.)
        print([head,mid,tail])
        paramList, freezeParamList = generateReverseParamsResnet18(net, initModel, finishedModel, layeredParams,
                                                                   filePath, strucName, datasetName, [mid])
        param = paramList[0]
        optimizer = getOptimizer(net, LR)
        model, _, forgetTestAcc = train(net, trainloader, testloader, forgetTestLoader, freezeParamList[0], pre_epoch,
                                        EPOCH, param, device, optimizer, criterion, filePath, logName, pthName)
        if forgetTestAcc > 1:
            forgetTestAcc /= 100
        if forgetTestAcc < accThreshold:
            tail = mid
        else:
            head = mid
    # 返回
    resp = {'message': "success", 'result': 'ok', 'data': 19-tail}
    print(resp.__str__())
    return HttpResponse(json.dumps(resp), content_type="application/json")


def predict(request):

    params = request.GET
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

    SAMPLE_SIZE = 10000

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
    # net = VGG('VGG16').to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    print("Waiting Test!")

    targetFile = 'resnet18_cifar10_forget_two_kinds_20210321_25_machine_1.pth'
    savedFiles = [
        # "net_reverse_reset_all_after_training.pth",
        # "net_reverse_reset_conv1_after_training.pth",
        # "net_reverse_reset_conv10_after_training.pth",
        # "net_reverse_reset_conv11_after_training.pth",
        # "net_reverse_reset_conv12_after_training.pth",
        # "net_reverse_reset_conv13_after_training.pth",
        # "net_reverse_reset_conv14_after_training.pth",
        # "net_reverse_reset_conv15_after_training.pth",
        # "net_reverse_reset_conv16_after_training.pth",
        # "net_reverse_reset_conv17_after_training.pth",
        # "net_reverse_reset_conv2_after_training.pth",
        # "net_reverse_reset_conv3_after_training.pth",
        # "net_reverse_reset_conv4_after_training.pth",
        # "net_reverse_reset_conv5_after_training.pth",
        # "net_reverse_reset_conv6_after_training.pth",
        # "net_reverse_reset_conv7_after_training.pth",
        # "net_reverse_reset_conv8_after_training.pth",
        # "net_reverse_reset_conv9_after_training.pth",
        # "resnet18_cifar10_fc_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv1_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv10_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv11_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv12_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv13_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv14_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv15_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv16_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv17_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv2_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv3_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv4_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_1_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_2_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_3_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_4_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_5_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_6_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_7_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_8_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_continuous_check_forget_9_kinds.pth",
        # "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv6_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv7_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv8_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_fc_conv9_before_training.pth_forget_two_kinds_after_finetuning_30_no_freezing_second.pth",
        # "resnet18_cifar10_forget_two_kinds_20210321_25_machine_1.pth",
        # "resnet18_cifar10_forget_two_kinds_30_29_time_.pth",
        # "resnet18_cifar10_forget_two_kinds_finished_saving_20210515_1.pth",
        # "resnet18_cifar10_forget_two_kinds_finished_saving_20210515_21.pth",
        # "resnet18_cifar10_forget_two_kinds_finished_saving_30_0_time_.pth",
        # "resnet18_cifar10_forget_two_kinds_finished_saving_30_1_time_.pth",
        # "resnet18_cifar10_forget_two_kinds_finished_saving_30_28_time_.pth",
        # "resnet18_cifar10_normal_train_finished_saving_60.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_1_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_2_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_3_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_4_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_5_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_6_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_7_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_8_kinds.pth",
        # "resnet18_cifar10_retrain_30_continuous_check_forget_9_kinds.pth",
        # "resnet18_cifar10_train_to_forget_two_kinds_finished_saving_20210515_11.pth",

        # "vgg16_cifar10_normal_train_finish_100_epochs.pth",
        # "vgg16_cifar10_reset_fc_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv1_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv10_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv11_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv12_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv13_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv2_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv3_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv4_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv6_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv7_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv8_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_reset_fc_conv9_before_training.pth_forget_two_kinds_after_finetuning_60_epochs.pth",
        # "vgg16_cifar10_retrain_forget_two_kinds_finished_60_epochs.pth",
        # "vgg16_cifar10_reverse_reset_conv1_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv10_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv11_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv12_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv13_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv14_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv2_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv3_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv4_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv5_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv6_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv7_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv8_after_training.pth",
        # "vgg16_cifar10_reverse_reset_conv9_after_training.pth",

        "resnet18_cifar10_fc_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv1_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv10_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv11_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv12_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv13_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv14_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv15_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv16_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv17_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv2_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv3_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv4_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv6_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv7_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv8_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
        "resnet18_cifar10_fc_conv9_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth",
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
    returnDictData = {}
    path = r'D:\ww2\defense-backend\model\\'

    queryFiles = []
    for i in fileList:
        queryFiles.append(savedFiles[int(i)])

    queryFiles = savedFiles
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
            returnDictData[file] = dataItem

        for i, file in enumerate(queryFiles, 0):
            print(file + '测试保留集分类准确率为：%.3f%%' % (100. * corrects[i] / totals[i]))

    end = datetime.datetime.now()
    resp = {}
    resp['message'] = str(end - start)
    resp['result'] = 'ok'
    resp['data'] = returnDictData
    print(end - start)
    return HttpResponse(json.dumps(resp), content_type="application/json")