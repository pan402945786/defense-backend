import pandas as pd
import csv
import os
import sys
sys.path.append("..")
# from common.resnet_100kinds_vggface2 import ResNet18
from common.resnet_1 import ResNet18
import torch
import shutil
import pickle
import time
import math

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    # df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df = pd.read_csv(identity_list, sep=',', quoting=csv.QUOTE_ALL, encoding="utf-8")
    # df["class"] = -1
    df["class"] = range(N_IDENTITY)
    # df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    # df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    # print(df)
    return id_label_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def generateParamsResnet18(net, former, later, layeredParams, isReverse, filePath, strucName, datasetName, layer):
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

            if isReverse:
                if i == len(layeredParams)-1:
                    continue
                for j in range(i+1):
                    freezeParams = freezeParams + layeredParams[len(layeredParams)-j-1]
                resetLayerName = "reverse_reset_" + str(i+1) + "_"
            else:
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


def generateReverseParamsResnet18(net, former, later, layeredParams, filePath, strucName, datasetName, layer):
    # 模型定义-ResNet
    # net = ResNet18().to(device)
    checkpoint = torch.load(filePath+later, map_location='cpu')
    net.load_state_dict(checkpoint)
    params = net.state_dict()
    toLoad = params
    checkpoint = torch.load(filePath+former, map_location='cpu')
    fileNameList = []
    freezeParamsList = []
    for i, params in enumerate(layeredParams):
        if i+1 in layer:
            newLayerParams = []
            for j in range(i+1):
                newLayerParams = newLayerParams + layeredParams[j]
            freezeParams = []
            # if i == len(layeredParams) - 1:
            #     continue
            for j in range(len(layeredParams)-i-1):
                freezeParams = freezeParams + layeredParams[len(layeredParams) - j - 1]
            resetLayerName = "reverse_reset_former_" + str(i + 1) + "_"
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

# param是参数文件名称
def trainFunc(net,device,trainloader,testloader,optimizer,criterion,scheduler,fileAccName,fileLogName,EPOCH,BATCH_SIZE,T_threshold,pre_epoch,param,args):
    # 训练
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    best_acc = 0
    tolerate = 10
    with open(fileAccName, "a+") as f:
        with open(fileLogName, "a+")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                lastLoss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.cuda()
                    labels = labels.cuda()
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
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), param,
                             optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.6f'
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('\n')
                    f2.flush()
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0.0
                    total = 0.0
                    sum_loss = 0
                    for i, data in enumerate(testloader):
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
                        lastLoss = sum_loss / (i + 1)
                    print('测试分类准确率为：%.3f%%, 当前学习率： %.3f, last loss: %.3f' % (
                    100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr'], lastLoss))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    if acc > best_acc:
                        best_acc = acc
                        print('Saving best acc model......')
                        torch.save(net.state_dict(), '%s/%s_best_acc_model.pth' % (
                            args.outf, param))
                        f.write("save best model\n")
                        f.flush()
                    if (epoch + 1) % 10 < 1:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/%s_%03d_epoch.pth' % (
                        args.outf, param.replace("before", "after"), epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f,BATCH_SIZE:%d,lastLoss:%.3f" % (
                        epoch + 1, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        optimizer.state_dict()['param_groups'][0]['lr'], BATCH_SIZE, lastLoss))
                    f.write('\n')
                    f.flush()
                scheduler.step(lastLoss, epoch=epoch)
                if lastTrainLoss < T_threshold and epoch > tolerate:
                    print('train loss达到限值%s，提前退出' % lastTrainLoss)
                    print('Saving model......')
                    torch.save(net.state_dict(),
                               '%s/%s_%03d_epoch.pth' % (args.outf, param.replace("before", "after"), epoch + 1))
                    f.write("train loss达到限值%s，提前退出" % lastTrainLoss)
                    f.write('\n')
                    f.flush()
                    break
                if optimizer.state_dict()['param_groups'][0]['lr'] < 0.003:
                    print("学习率过小，退出")
                    f.write("学习率过小，退出")
                    f.write('\n')
                    f.flush()
                    break
            print('Saving model......')
            torch.save(net.state_dict(),
                       '%s/%s_%03d_epoch.pth' % (args.outf, param.replace("before", "after"), epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


def calcForgetPercent(forgetArr, retainArr, forgetRetrain, retainRetrain):
    length = len(forgetArr)
    forgetPercent = []
    denominator = math.sqrt(pow(forgetRetrain, 2) + pow(retainRetrain, 2))
    for i in range(length):
        if forgetArr[i] > 1:
            forgetArr[i] = forgetArr[i] / 100
        if retainArr[i] > 1:
            retainArr[i] = retainArr[i] / 100

        numerator = math.sqrt(pow(forgetArr[i]-forgetRetrain, 2) + pow(retainArr[i]-retainRetrain, 2))
        forgetPercent.append(1 - numerator / denominator)
    return forgetPercent


def calcSingleForgetPercent(forgetAcc, retainAcc, forgetRetrain, retainRetrain):
    if forgetAcc > 1:
        forgetAcc = forgetAcc / 100
    if retainAcc > 1:
        retainAcc = retainAcc / 100
    if forgetRetrain > 1:
        forgetRetrain = forgetRetrain / 100
    if retainRetrain > 1:
        retainRetrain = retainRetrain / 100
    numerator = math.sqrt(pow(forgetAcc-forgetRetrain, 2) + pow(retainAcc-retainRetrain, 2))
    denominator = math.sqrt(pow(forgetRetrain, 2) + pow(retainRetrain, 2))
    return 1 - numerator / denominator

