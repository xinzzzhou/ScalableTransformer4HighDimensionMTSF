import numpy as np


def RMSSE(true, pred, train=None): 
    n = len(train)
    numerator = np.mean(np.square(true - pred))
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    return np.sqrt(numerator/denominator)

def WRMSPE(pred, true):
    numerator = np.sqrt(np.mean((pred - true) ** 2))
    denominator = np.mean(np.abs(true))
    return numerator/denominator

def sMAPE(pred, true, constant=200):
    return np.mean(constant * np.abs(pred - true) / (np.abs(pred) + np.abs(true)))

def WAPE(pred, true):
    numerator = np.sum(np.abs(pred - true))
    denominator = np.sum(np.abs(true))
    return numerator/denominator

def RMSSE(true, pred, train=None, seasonality=12): 
    n = len(train)
    numerator = np.mean(np.square(true - pred))
    denominator = 1/(n-1)*np.sum(np.square((train[seasonality:] - train[:-seasonality])))
    return np.sqrt(numerator/denominator)

def MASE(pred, true, train=None, seasonality=12):
    n = len(train)
    numerator = np.mean(np.abs(pred - true))    
    denominator = 1/(n-1)*np.sum(np.abs((train[seasonality:] - train[:-seasonality])))
    return numerator/denominator

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, train=None, seasonality=6):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    rmsse = RMSSE(true, pred, train, seasonality) 
    wrmspe = WRMSPE(pred, true)
    smape = sMAPE(pred, true)
    wape = WAPE(pred, true)
    mase = MASE(pred, true, train, seasonality)
    return mae, mse, rmse, mape, mspe, rse, corr, rmsse, wrmspe, smape, wape, mase
