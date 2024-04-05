import sys
import csv
import time
import xlrd
from scipy.io import arff
import numpy as np


def data_update(data_before, weight):
    Data_update = []
    func = lambda x, y: x * y
    for i in range(len(data_before)):
        result = map(func, data_before[i], weight)
        result = list(result)
        Data_update.append(result)
    return Data_update


def cal_aibi(all_cluster, update_data, center):
    # ai2grad用于存储所有数据点的ai对所有特征权重的偏导
    ai2grad = []
    # bi2grad用于存储所有数据点的bi对所有特征权重的偏导
    bi2grad = []
    # ai记录所有数据点的ai值
    ai = []
    # bi记录所有数据点的bi值
    bi = []
    for i in range(len(all_cluster)):
        for j in range(len(all_cluster[i])):
            # 记录数据点与异簇簇中心的距离
            dist2Center = []
            for k in range(len(center)):
                dist = np.linalg.norm(np.array(update_data[all_cluster[i][j]]) - np.array(center[k]))
                if i == k:
                    ai.append(dist)
                    tempAiGrad = 0.5 * (np.array(update_data[all_cluster[i][j]]) - np.array(center[k])) / dist
                    ai2grad.append(tempAiGrad)
                else:
                    dist2Center.append(dist)
            # NearestIndex：距离数据点最近的异簇索引
            NearestIndex = dist2Center.index(min(np.array(dist2Center)))
            bi.append(min(dist2Center))
            tempBiGrad = 0.5 * (np.array(update_data[all_cluster[i][j]]) - np.array(center[NearestIndex])) / min(dist2Center)
            bi2grad.append(tempBiGrad)
    return ai2grad, ai, bi2grad, bi


def WKBSC(X, K, n, floss_min, floss_max, max_iters=300):
    loss_ratio_record = []
    X = np.array(X)
    # 初始化权重
    weight = [1/len(X[0]) for _ in range(len(X[0]))]
    # 随机选择K个初始簇中心
    centroids = X[np.random.choice(range(len(X)), K, replace=False)]
    labels = []
    Floss = []
    while True:
        # 拟合数据
        data = np.array(data_update(list(X), weight))
        # 更新label、center、cluster_indices
        for _ in range(max_iters):
            # 分配数据点到最近的簇
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)
            labels = np.argmin(distances, axis=-1)
            # 更新簇中心
            new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(K)])
            # 如果簇中心不再发生变化，退出迭代
            if np.all(centroids == new_centroids):
                centroids = new_centroids
                break
            # 更新簇中心
            centroids = new_centroids
        #  cluster_indices记录每个簇包含的数据点的数量
        cluster_indices = [np.where(labels == k)[0] for k in range(K)]
        weight_ai, ai, weight_bi, bi = cal_aibi(cluster_indices, data, centroids)

        weight_change = np.array([0 for _ in range(len(X[0]))])
        for l_ in range(len(X[0])):
            for _l in range(len(data)):
                a = weight_ai[_l][l_] / bi[_l]
                b = weight_bi[_l][l_] * ai[_l] / bi[_l] ** 2
                if np.isnan(a) or np.isnan(b):
                    continue
                weight_change[l_] = weight_change[l_] - a + b
        weight_change = weight_change / len(data)
        for k in range(len(weight)):
            weight[k] = weight[k] - n * weight_change[k]
        f_loss = sum(np.array(ai)) - sum(np.array(bi)) / len(data)
        Floss.append(f_loss)

        if len(Floss) > 1:
            new_floss = len(Floss) - 1
            last_Floss = new_floss - 1
            loss_ratio_record.append(Floss[new_floss] / Floss[last_Floss])
            if floss_min < Floss[new_floss] / Floss[last_Floss] < floss_max:
                return labels


if __name__ == '__main__':

    n = 0.01
    floss_min = 0.999
    floss_max = 1.001

    data = []
    # data = ...
    label_true = []
    # label_true = ...

    X = np.array(data)
    K = len(np.unique(np.array(label_true)))
    labels = WKBSC(X, K, n, floss_min, floss_max)






