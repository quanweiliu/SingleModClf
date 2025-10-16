import os
# copy image
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from timm.loss import LabelSmoothingCrossEntropy
from thop import profile, clever_format

from model import basic_cnn, resNet18, resNet50, InceptionV3
from model import mobileNetV2, mobileNetV3, mobileNetV1, mobileNetV1_M
from metrics.metrics_v1 import metric_log
from utils.trainer import fit
from utils.tester import test
from utils.options import get_args

args = get_args()
specises = ['Absent', 'Present']
num_classes = len(specises)
BATCHSZ = 16

# model_name = "basicCnnNet1"
# model_name = "basicCnnNet2"
# model_name = 'mobileNetV2'
# model_name = 'mobileNetV3'
# model_name = 'mobilenetv3_large'
# model_name = 'mobilenetv3_small'
# model_name = 'mobileNetV1'
model_name = 'resNet18'
# model_name = 'resNet50'
# model_name = 'InceptionV3'

# result_dir = "/home/icclab/Documents/lqw/Multimodal_Classification/SingleModClf/results/1015-144336-mobilenetv3_small"
result_dir = "/home/icclab/Documents/lqw/Multimodal_Classification/SingleModClf/results/1015-152720-resNet18"
# result_dir = "/home/icclab/Documents/lqw/Multimodal_Classification/SingleModClf/results/1016-110255-resNet50"

base_dir = "/home/icclab/Documents/lqw/DatasetSMD/DidemnumPerlucidum"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

for train_or_test in ['train', 'test']:
    for spec in specises:
        print(train_or_test, spec, len(os.listdir(os.path.join(base_dir, train_or_test, spec))))

transform = transforms.Compose([
    # transforms.Resize((156, 156)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),   # 三个作用，归一化，channel first， tensor
    # RGB 三个维度
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # 这里的均值方差是猜的，知道的情况可以用正确的
])

# train
train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_ds = torchvision.datasets.ImageFolder(test_dir, transform=transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCHSZ, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCHSZ)

id_to_class = dict((v, k) for k, v in train_ds.class_to_idx.items())
id_to_class


if model_name == "basicCnnNet1":
    model = basic_cnn.Net1(classes=num_classes).to(args.device)
elif model_name == "basicCnnNet2":
    model = basic_cnn.Net2(classes=num_classes).to(args.device)
elif model_name =="mobileNetV1":
    model = mobileNetV1.MyMobileNet_v1(num_classes=num_classes).to(args.device)
elif model_name =="mobileNetV1_M":
    model = mobileNetV1_M.MyMobileNet_v1_M(width_multiplier=1, num_classes=num_classes).to(args.device)
elif model_name =="mobileNetV2":
    model = mobileNetV2.mobilenet_v2(width_mult=1, classes=num_classes).to(args.device)
elif model_name =="mobilenetv3_small":
    model = mobileNetV3.mobilenetv3_small(num_classes=num_classes).to(args.device)
elif model_name =="mobilenetv3_large":
    model = mobileNetV3.mobilenetv3_large(num_classes=num_classes).to(args.device)
elif model_name =="resNet18":
    model = resNet18.resNet18(bands=3, classes=num_classes).to(args.device)
elif model_name =="resNet50":
    model = resNet50.resNet50(bands=3, classes=num_classes).to(args.device)
elif model_name =="InceptionV3":
    # model = InceptionV3.MyInception_v3(num_classes=9).to(args.device)
    model = InceptionV3.GoogLeNetV3(num_classes=num_classes).to(args.device)
print(model_name)


loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
print(model_name)

path = os.path.join(result_dir, 'weights.pth')
if path != '':
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'], strict=True)
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(path), "start", epoch_start)


starttime = time.time()
test_loss, test_acc, test_preds, test_labels = test(model, test_dl, loss_fn, args)
test_time = time.time() - starttime


matrix, classification, accuracy, precision, recall, f1, kappa = \
                            metric_log(test_preds, test_labels)
f = open(os.path.join(result_dir, 'results.txt'), 'a+')
str_results = '\n ======================' \
            + '\n' + classification \
            + "\nmatrix = \t\t" + str(matrix) \
            + "\nepoch = \t" + str(epoch_start) \
            + "\naccuracy = \t\t" + str(round(accuracy, 4)) \
            + "\nprecision = \t" + str(round(precision, 4)) \
            + "\nrecall = \t\t" + str(round(recall, 4)) \
            + "\nf1 = \t\t\t\t" + str(round(f1, 4)) \
            + "\nkappa = \t\t\t" + str(round(kappa, 4)) \
            + '\ntest time = \t' + str(round(test_time, 2)) \
            + '\n'
f.write(str_results)
f.close()
















