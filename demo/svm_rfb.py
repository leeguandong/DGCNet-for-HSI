import argparse
import collections
import datetime
import time

import numpy as np
import torch
from sklearn import preprocessing, metrics

from thyperspectral import load_dataset, sampling, aa_and_each_accuracy, record_output, extract_samll_cubic, svm_rbf, \
    list_to_colormap, classification_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="data_process")
    parser.add_argument("--dataset", type=str, default="Indian",
                        choices=["Indian", "PaviaU", "Pavia", "KSC", "Botswana", "Salinas"])

    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--path_length", type=int, default=3)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--saved", type=str, default='./saved')

    args = parser.parse_args()
    return args


seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
args = parse_args()
PATCH_LENGTH = args.path_length
ITER = args.iter
saved = args.saved

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(args)
print('The size of the HSI data is:', data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    # x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    # x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    # x_all = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2] * INPUT_DIMENSION)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2] * INPUT_DIMENSION)
    x_all = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2] * INPUT_DIMENSION)

    tic1 = time.clock()
    net = svm_rbf(x_train, y_train).train()

    toc1 = time.clock()

    pred_test = []
    tic2 = time.clock()
    with torch.no_grad():
        for i in range(x_test.shape[0]):
            pred_test.extend(net.predict(x_test[i]))
    toc2 = time.clock()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1

    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)

    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc

print("--------" + net.name + " Training Finished-----------")
record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
              'records/' + net.name + day_str + '_' + args.dataset + 'split：' + str(VALIDATION_SPLIT) + '.txt')

pred_test = []
for i in range(x_all.shape[0]):
    pred_test.extend(np.array(net.predict(x_all[i])))

gt = gt_hsi.flatten()
x_label = np.zeros(gt.shape)
for i in range(len(gt)):
    if gt[i] == 0:
        gt[i] = 17
        # x[i] = 16
        x_label[i] = 16
        # else:
        #     x_label[i] = pred_test[label_list]
        #     label_list += 1
gt = gt[:] - 1
x_label[total_indices] = pred_test
x = np.ravel(x_label)

# print('-------Save the result in mat format--------')
# x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
# sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

y_list = list_to_colormap(x)
y_gt = list_to_colormap(gt)

y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

path = saved + '/' + net.name
classification_map(y_re, gt_hsi, 300, path + '/classification_maps/' + args.dataset + '_' + net.name + '.png')
classification_map(gt_re, gt_hsi, 300, path + '/classification_maps/' + args.dataset + '_gt.png')
print('------Get classification maps successful-------')
