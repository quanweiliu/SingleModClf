import os
import csv
import torch
import time
from tqdm import tqdm

def test(model, testloader, loss_fn, args):

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    test_pred_list = []
    test_label_list = []

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

            test_label_list.extend(y.cpu().numpy())
            test_pred_list.extend(y_pred.cpu().numpy())

    test_loss = test_running_loss / len(testloader.dataset)
    test_accuracy = test_correct / test_total

    print('test_loss: ', round(test_loss, 3),
        'test_accuracy', round(test_accuracy, 3))

    return test_loss, test_accuracy, test_pred_list, test_label_list