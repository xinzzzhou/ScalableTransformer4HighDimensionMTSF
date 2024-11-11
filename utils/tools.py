import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import datetime

plt.switch_backend('agg')

def print_with_timestamp(*args, **kwargs):
    # get current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print time
    print(f"[{current_time}]", *args, **kwargs)
    
def visual_loss(x, ys, labels, path):
    """
    Loss visualization
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(path+'.pdf', bbox_inches='tight')
    

def result_print(input_string):
    # for overleaf
    split_values = input_string.split(',')
    formatted_values = []

    for value in split_values:
        numeric_part = value.split(':')[-1].strip()
        formatted_value = "{:.4f}".format(float(numeric_part))
        formatted_values.append(formatted_value)
    output_string = ", ".join(formatted_values)
    print(output_string)
    return output_string
    
    
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lr_adj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lr_adj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    #
    elif args.lr_adj == 'type3':
        lr_adjust = {
            5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
            15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
            25: args.learning_rate * 0.5 ** 5
        }
    elif args.lr_adj == 'type4':
        lr_adjust = {
            2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
            6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
            10: args.learning_rate * 0.5 ** 5
        }
    elif args.lr_adj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lr_adj == 'type6':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lr_adj == 'type7':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lr_adj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    
    # elif args.lr_adj == 'TST':
    #     lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        #
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))
        args.logger.info('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)