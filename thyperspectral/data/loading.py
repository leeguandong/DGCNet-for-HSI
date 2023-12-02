import math
import os
from operator import truediv

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data

from thyperspectral.utils import extract_samll_cubic

pwd = os.getcwd().split("demo")[0]


def load_dataset(args):
    Dataset = args.dataset
    if Dataset == 'Indian':
        mat_data = sio.loadmat(os.path.join(pwd, 'dataset/IN/Indian_pines_corrected.mat'))
        mat_gt = sio.loadmat(os.path.join(pwd, 'dataset/IN/Indian_pines_gt.mat'))
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        TRAIN_SPLIT = args.train_split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * TRAIN_SPLIT)

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat(os.path.join(pwd, 'dataset/UP/PaviaU.mat'))
        gt_uPavia = sio.loadmat(os.path.join(pwd, 'dataset/UP/PaviaU_gt.mat'))
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        TRAIN_SPLIT = args.train_split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * TRAIN_SPLIT)

    if Dataset == 'Pavia':
        uPavia = sio.loadmat(os.path.join(pwd, 'dataset/Pavia/Pavia.mat'))
        gt_uPavia = sio.loadmat(os.path.join(pwd, 'dataset/Pavia/Pavia_gt.mat'))
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        TRAIN_SPLIT = args.train_split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * TRAIN_SPLIT)

    if Dataset == 'Salinas':
        SV = sio.loadmat(os.path.join(pwd, 'dataset/Salinas/Salinas_corrected.mat'))
        gt_SV = sio.loadmat(os.path.join(pwd, 'dataset/Salinas/Salinas_gt.mat'))
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        TRAIN_SPLIT = args.train_split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * TRAIN_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat(os.path.join(pwd, 'dataset/KSC/KSC.mat'))
        gt_KSC = sio.loadmat(os.path.join(pwd, 'dataset/KSC/KSC_gt.mat'))
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        TRAIN_SPLIT = args.train_split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * TRAIN_SPLIT)

    if Dataset == 'Botswana':
        BS = sio.loadmat(os.path.join(pwd, 'dataset/Botswana/Botswana.mat'))
        gt_BS = sio.loadmat(os.path.join(pwd, 'dataset/Botswana/Botswana_gt.mat'))
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        TRAIN_SPLIT = args.train_split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * TRAIN_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, TRAIN_SPLIT


def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        train[i] = indexes[:-nb_val]
        test[i] = indexes[-nb_val:]
        # train[i] = indexes[:nb_val]
        # test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y



