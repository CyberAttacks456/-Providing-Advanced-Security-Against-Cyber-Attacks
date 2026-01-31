import numpy as np
import math


def error_evaluation(sp, act):
    r = act
    x = sp
    points = np.zeros((1, x.shape[0]))
    abs_r = np.zeros((1, x.shape[0]))
    abs_x = np.zeros((1, x.shape[0]))
    abs_r_x = np.zeros((1, x.shape[0]))
    abs_x_r = np.zeros((1, x.shape[0]))
    absr_x__r = np.zeros((1, x.shape[0]))
    for j in range(1, x.shape[0]):
        points[j][0] = abs(x[0][j] - x[j-1][0])
    for i in range(len(r[0])):
        abs_r[0, i] = abs(r[0][i])
    for i in range(len(r[0])):
        abs_x[0, i] = abs(x[0][i])
    for i in range(len(r[0])):
        abs_r_x[0, i] = abs(r[0][i] - x[0][i])
    for i in range(len(r[0])):
        abs_x_r[0, i] = abs(x[0][i] - r[0][i])
    for i in range(len(r[0])):
        absr_x__r[0, i] = abs((r[0][i] - x[0][i]) / r[0][i])
    md = (100/len(x[0])) * sum(absr_x__r[0])
    smape = (1/len(x[0])) * sum(abs_r_x[0]/((abs_r[0] + abs_x[0]) / 2))
    mase = sum(abs_r_x)/((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x[0]) / len(r[0])
    rmse = (sum(abs_x_r[0] ** 2) / len(r[0])) ** 0.5
    onenorm = sum(abs_r_x[0])
    twonorm = (sum(abs_r_x[0] ** 2) ** 0.5)
    infinitynorm = max(abs_r_x[0])
    EVAL_ERR = [md, smape, mase, mae, rmse, onenorm, twonorm, infinitynorm]
    return EVAL_ERR

