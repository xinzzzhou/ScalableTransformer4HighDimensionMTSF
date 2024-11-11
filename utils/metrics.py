import numpy as np


def WRMSPE(pred, true):
    numerator = np.sqrt(np.mean((pred - true) ** 2))
    denominator = np.mean(np.abs(true))
    return numerator/denominator

def WAPE(pred, true):
    numerator = np.sum(np.abs(pred - true))
    denominator = np.sum(np.abs(true))
    return numerator/denominator

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def metric(pred, true, train=None, seasonality=6):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    wrmspe = WRMSPE(pred, true)
    wape = WAPE(pred, true)
    return rmse, wrmspe, mae, wape
