import os
import csv
import torch
import time
from tqdm import tqdm


def fit(epoch, model, trainloader, testloader, optim, loss_fn, args):
    correct = 0
    total = 0
    running_loss = 0
    starttime = time.time()

    # 训练模式影响 dropout，Batch Normalization
    model.train()
    for x, y in tqdm(trainloader):
        x, y = x.to(args.device), y.to(args.device)
        y_pred = model(x)
        # print("y_pred", y_pred, "y", y.shape)
        # print("y_pred", y_pred.shape, "y", y.shape)


        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
        
    # 每一个样本的平均 loss 和 acc
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_accuracy = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    endtime = time.time()
    train_time = endtime - starttime

    # 预测模式影响 dropout Batch, Normalization
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(testloader):
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_accuracy = test_correct / test_total
    # test_time = time.time() - endtime

    print('epoch', epoch, 
            'loss: ', round(epoch_loss, 6),
            'accuracy: ', round(epoch_accuracy, 4),
            'train_time: ', round(train_time, 2),
            'test_loss: ', round(epoch_test_loss, 6),
            'test_accuracy: ', round(epoch_test_accuracy, 4),
            # 'test_time: ', round(test_time, 2),
            )
    
    with open(os.path.join(args.result_dir, "log.csv"), 'a+', encoding='gbk') as f:
        row=[["epoch", epoch,
            "loss", round(epoch_loss, 6),
            "accuracy", round(epoch_accuracy, 4),
            "train_time", round(train_time, 2),
            "test_loss", round(epoch_test_loss, 6),
            "test_accuracy", round(epoch_test_accuracy, 4),
            # 'test_time: ', round(test_time, 2),
            '\n']]
        write=csv.writer(f)
        for i in range(len(row)):
            write.writerow(row[i])


    return epoch_loss, epoch_accuracy, epoch_test_loss, epoch_test_accuracy, train_time