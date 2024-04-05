import sys
import numpy as np


def data_update(data_before, weight):
    Data_update = []
    func = lambda x, y: x * y
    for i in range(len(data_before)):
        result = map(func, data_before[i], weight)
        result = list(result)
        Data_update.append(result)
    return Data_update


def cal_ai(length, all_cluster, update_data):
    ai = []
    # all_ai_w用于存储所有簇计算的ai对权重的求导
    all_ai_w = [[] for _ in range(length)]
    for i in range(len(all_cluster)):
        cluster_w = [[] for _ in range(length)]
        if len(all_cluster[i]) == 1:
            ai.append(0)
        else:
            for j in range(0, len(all_cluster[i])):
                # point_w用于记录单个数据点与其他所有同簇数据点之间的欧式距离对权重的求导
                point_w = [[] for _ in range(length)]
                ai_temp = []
                for k in range(0, len(all_cluster[i])):
                    # 计算同簇中两个数据点之间的欧氏距离
                    a = update_data[all_cluster[i][j]]
                    b = update_data[all_cluster[i][k]]
                    dist = np.linalg.norm(np.array(a) - np.array(b))
                    if dist != 0:
                        ai_temp.append(dist)
                    # tempGrad 是单个数据点和同簇中单个数据点之间的欧氏距离对权重的倒数
                    if dist == 0:
                        tempGrad = np.array([0] * len(a))
                    else:
                        tempGrad = 0.5 * (np.array(a) - np.array(b)) / dist
                    for ll in range(len(tempGrad)):
                        point_w[ll].append(tempGrad[ll])
                # ai存储的是所有数据点的ai值
                ai.append(sum(np.array(ai_temp)) / len(ai_temp))
                for k in range(len(point_w)):
                    # 此时cluster_w中存储的是簇中所有单个数据点与同簇数据点之间的欧氏距离对各个维度权重的求导的均值
                    cluster_w[k].append(sum(np.array(point_w[k])) / len(point_w[k]))
        # 从簇的角度记录所有簇中数据点的ai对不同数据维度的权重求导
        for l_ in range(len(cluster_w)):
            if len(cluster_w[l_]) == 0:
                all_ai_w[l_].append(0)
            else:
                all_ai_w[l_].append(sum(np.array(cluster_w[l_])) / len(cluster_w[l_]))

    update_ai_w = []
    for i in range(len(all_ai_w)):
        total = np.sum(np.array(all_ai_w[i]))
        total = total / len(all_ai_w[i])
        update_ai_w.append(total)
    return update_ai_w, ai


def cal_bi(length, all_cluster, update_data, center):
    # point_w用于记录所有数据点的bi值对w的求导
    point_w = [[] for _ in range(length)]
    # bi记录所有数据点的bi值
    bi = []
    for i in range(len(all_cluster)):
        for j in range(len(all_cluster[i])):
            tempdist = []
            for k in range(len(center)):
                if i == k:
                    continue
                else:
                    dist = np.linalg.norm(np.array(update_data[all_cluster[i][j]]) - np.array(center[k]))
                    tempdist.append(dist)

            center_index = tempdist.index(min(tempdist))
            a = update_data[all_cluster[i][j]]
            b = center[center_index]
            temp_dist = np.linalg.norm(np.array(a) - np.array(b))
            bi.append(temp_dist)
            tempGrad = 0.5 * (np.array(a) - np.array(b)) / temp_dist
            for l in range(length):
                point_w[l].append(tempGrad[l])

    update_bi_w = []
    for i in range(len(point_w)):
        total = np.sum(np.array(point_w[i]))
        total = total / len(point_w[i])
        update_bi_w.append(total)
    return update_bi_w, bi


def WKBSC(X, K, n, floss_min, floss_max, max_iters=100):
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
        weight_ai, ai = cal_ai(len(X[0]), cluster_indices, data)
        weight_bi, bi = cal_bi(len(X[0]), cluster_indices, data, centroids)
        for k in range(len(weight)):
            weight[k] = weight[k] - n * (weight_ai[k] - weight_bi[k])
        f_loss = sum(np.array(ai) - np.array(bi)) / len(ai)
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
