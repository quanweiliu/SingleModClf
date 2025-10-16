from torch import nn
import torch.nn.functional as F


class Net1(nn.Module):
    '''
    Net 的优化版本1, 加入了 dropout
    '''
    def __init__(self, classes):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.dropout2d = nn.Dropout2d(0.5)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64*17*17, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2d(x)
        # print(x.size())
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        # dropout 抑制过拟合
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

class Net2(nn.Module):
    '''
    Net 的优化版本2, 加入了 BatchNorm
    '''
    def __init__(self, classes):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        # self.dropout2d = nn.Dropout2d(0.5)
        # self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64*17*17, 1024)
        self.bn_f1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn_f2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn3(x)
        # x = self.dropout2d(x)
        # print(x.size())
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        # dropout 抑制过拟合
        x = F.relu(self.fc1(x))
        x = self.bn_f1(x)
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn_f2(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x