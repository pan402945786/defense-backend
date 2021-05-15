from sklearn.svm import SVC
import torch
# import tqdm
from tqdm.autonotebook import tqdm
from sklearn.linear_model import LogisticRegression

import numpy as np
import torch.nn.functional as F
from resnet_1 import ResNet18
# from common.utils import *
from utils_1 import *
import os
import errno
import argparse
import torchvision.transforms as transforms
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

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
# trainset = torchvision.datasets.CIFAR10(root='D:\www\graduate_expriment\data', train=True, download=False, transform=transform_train) #训练数据集
# testset = torchvision.datasets.CIFAR10(root='D:\www\graduate_expriment\data', train=False, download=False, transform=transform_test)
print(len(trainset))
print(len(testset))

# 输入要删除的类别
forgetClasses = [8, 9]
forgottenExamples_train = []
unforgottenExamples_train = []
forgottenExamples_test = []
unforgottenExamples_test = []
for i, item in enumerate(trainset):
    # if i > 10:
    #     break
    if item[1] in forgetClasses:
        forgottenExamples_train.append(item)
    else:
        unforgottenExamples_train.append(item)
for i, item in enumerate(testset):
    # if i > 10:
    #     break
    if item[1] in forgetClasses:
        forgottenExamples_test.append(item)
    else:
        unforgottenExamples_test.append(item)

temp_test = []
for i, item in enumerate(testset):
    if i > 10:
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
# root_dir = r'/media/public/ml/resnet18_cifar10/model/'
root_dir = r'/home/ubuntu/ml/resnet18_cifar10/model/'
# root_dir = r'D:\www\graduate_expriment\resnet18_cifar10\model\\'
model = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_normal_train_finished_saving_60.pth")
model.load_state_dict(checkpoint)

modelf = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fisher_forget_model_1.pth")
modelf.load_state_dict(checkpoint)

model0 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_forget_two_kinds_finished_saving_30_29_time_.pth")
model0.load_state_dict(checkpoint)

model01_fc = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc.load_state_dict(checkpoint)

model01_fc_conv1 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv1_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv1.load_state_dict(checkpoint)

model01_fc_conv2 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv2_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv2.load_state_dict(checkpoint)

model01_fc_conv3 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv3_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv3.load_state_dict(checkpoint)

model01_fc_conv4 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv4_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv4.load_state_dict(checkpoint)

model01_fc_conv5 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv5_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv5.load_state_dict(checkpoint)

model01_fc_conv6 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv6_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv6.load_state_dict(checkpoint)

model01_fc_conv7 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv7_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv7.load_state_dict(checkpoint)

model01_fc_conv8 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv8_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv8.load_state_dict(checkpoint)

model01_fc_conv9 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv9_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv9.load_state_dict(checkpoint)

model01_fc_conv10 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv10_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv10.load_state_dict(checkpoint)

model01_fc_conv11 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv11_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv11.load_state_dict(checkpoint)

model01_fc_conv12 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv12_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv12.load_state_dict(checkpoint)

model01_fc_conv13 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv13_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv13.load_state_dict(checkpoint)

model01_fc_conv14 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv14_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv14.load_state_dict(checkpoint)

model01_fc_conv15 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv15_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv15.load_state_dict(checkpoint)

model01_fc_conv16 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv16_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv16.load_state_dict(checkpoint)

model01_fc_conv17 = ResNet18().to(device)
checkpoint = torch.load(root_dir + "resnet18_cifar10_fc_conv17_before_training.pth_forget_two_kinds_after_finetuning_30_second_time.pth")
model01_fc_conv17.load_state_dict(checkpoint)


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False)
    prob = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    # clf = SVC(C=3, gamma='auto', kernel='rbf')
    clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()


# def plot_entropy_dist(model, ax, title):
#     train_loader_full, test_loader_full = datasets.get_loaders(dataset, batch_size=100, seed=0, augment=False,
#                                                                shuffle=False)
#     indexes = np.flatnonzero(np.array(train_loader_full.dataset.targets) == class_to_forget)
#     replaced = np.random.RandomState(0).choice(indexes, size=100 if num_to_forget == 100 else len(indexes),
#                                                replace=False)
#     X_f, Y_f, X_r, Y_r = get_membership_attack_data(train_loader_full, test_loader_full, model, replaced)
#     sns.distplot(np.log(X_r[Y_r == 1]).reshape(-1), kde=False, norm_hist=True, rug=False, label='retain', ax=ax)
#     sns.distplot(np.log(X_r[Y_r == 0]).reshape(-1), kde=False, norm_hist=True, rug=False, label='test', ax=ax)
#     sns.distplot(np.log(X_f).reshape(-1), kde=False, norm_hist=True, rug=False, label='forget', ax=ax)
#     ax.legend(prop={'size': 14})
#     ax.tick_params(labelsize=12)
#     ax.set_title(title, size=18)
#     ax.set_xlabel('Log of Entropy', size=14)
#     ax.set_ylim(0, 0.4)
#     ax.set_xlim(-35, 2)


def membership_attack(retain_loader, forget_loader, test_loader, model):
    prob = get_membership_attack_prob(retain_loader, forget_loader, test_loader, model)
    print("Attack prob: ", prob)
    return prob


# %%
attack_dict = {}
# %%
# print('original')
# attack_dict['Original'] = membership_attack(retain_loader, forget_loader, test_loader_full, model)
# # %%
# print('Retrain')
# attack_dict['Retrain'] = membership_attack(retain_loader, forget_loader, test_loader_full, model0)
# %%
# print('Fisher')
# attack_dict['Fisher'] = membership_attack(retain_loader, forget_loader, test_loader_full, modelf)
# print('ft_fc')
# attack_dict['ft_fc'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc)
# print('ft_fc_conv1')
# attack_dict['ft_fc_1'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv1)
# print('ft_fc_conv2')
# attack_dict['ft_fc_2'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv2)
# print('ft_fc_conv3')
# attack_dict['ft_fc_3'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv3)
# print('ft_fc_conv4')
# attack_dict['ft_fc_4'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv4)
# print('ft_fc_conv5')
# attack_dict['ft_fc_5'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv5)
# print('ft_fc_conv6')
# attack_dict['ft_fc_6'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv6)
# print('ft_fc_conv7')
# attack_dict['ft_fc_7'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv7)
# print('ft_fc_conv8')
# attack_dict['ft_fc_8'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv8)
# print('ft_fc_conv9')
# attack_dict['ft_fc_9'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv9)
# print('ft_fc_conv10')
# attack_dict['ft_fc_10'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv10)
print('ft_fc_conv11')
attack_dict['ft_fc_11'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv11)
print('ft_fc_conv12')
attack_dict['ft_fc_12'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv12)
print('ft_fc_conv13')
attack_dict['ft_fc_13'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv13)
print('ft_fc_conv14')
attack_dict['ft_fc_14'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv14)
print('ft_fc_conv15')
attack_dict['ft_fc_15'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv15)
print('ft_fc_conv16')
attack_dict['ft_fc_16'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv16)
print('ft_fc_conv17')
attack_dict['ft_fc_17'] = membership_attack(retain_loader, forget_loader, test_loader_full, model01_fc_conv17)


