import sys

sys.path.append("/home/ivms/local_disk/TpHyperspectralcls")

import os
import argparse
import collections
import datetime
import time

import numpy as np
import torch
from sklearn import preprocessing, metrics
from torch import optim

from thyperspectral import SSRN_network, FDSSC_network, DBMA_network, CDCNN_network, DBDA_network_MISH, FeatherNet_network, DydenseNet, \
    load_dataset, sampling, generate_iter, aa_and_each_accuracy, record_output, generate_png,FlgcdenseNet
from tools import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_list = {"ssrn": SSRN_network, 'fdssc': FDSSC_network, 'dbma': DBMA_network, 'cdcnn': CDCNN_network,
            'dbda': DBDA_network_MISH, 'feathernet3d': FeatherNet_network, 'dydenseNet': DydenseNet,'flgcdenseNet':FlgcdenseNet}


def parse_args():
    parser = argparse.ArgumentParser(description="data_process")
    parser.add_argument("--dataset", type=str, default="KSC",
                        choices=["Indian", "PaviaU", "Pavia", "KSC", "Botswana", "Salinas"])
    parser.add_argument("--model", type=str, default="dydenseNet",
                        choices=["ssrn", 'fdssc', 'dbma', 'cdcnn', 'dbda', 'feathernet3d', 'dydenseNet'])

    parser.add_argument("--iter", type=int, default=4)
    parser.add_argument("--path_length", type=int, default=3)
    parser.add_argument("--train_split", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=80)

    parser.add_argument("--saved", type=str, default='/home/ivms/local_disk/saved/')

    args = parser.parse_args()
    return args


def main():
    for PATCH_LENGTH in [ 7, 8]:
        seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]

        args = parse_args()
        # PATCH_LENGTH = args.path_length
        ITER = args.iter
        lr = args.lr
        batch_size = args.batch_size
        num_epochs = args.num_epochs
        saved = args.saved + f"/_{str(args.train_split)}_{str(2 * PATCH_LENGTH + 1)}_{args.dataset}"

        if not os.path.exists(saved):
            os.makedirs(saved)

        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')

        print('-----Importing Dataset-----')
        data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, TRAIN_SPLIT = load_dataset(args)
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
            net = net_list[args.model](BAND, CLASSES_NUM)
            net = torch.nn.DataParallel(net).cuda()
#             import pdb;pdb.set_trace()

            loss = torch.nn.CrossEntropyLoss().cuda()
            optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0001)
            # optimizer = optim.SGD(net.parameters(),lr=lr,weight_decay=0.0005)
            # optimizer = optim.RMSprop(net.parameters(),lr=lr)
            # optimizer = optim.Adagrad(net.parameters(),lr=lr,weight_decay=0.01)
            # optimizer = optim.Adadelta(net.parameters(),lr=lr)

            np.random.seed(seeds[index_iter])
            train_indices, test_indices = sampling(TRAIN_SPLIT, gt)
            _, total_indices = sampling(1, gt)

            TRAIN_SIZE = len(train_indices)
            print('Train size: ', TRAIN_SIZE)
            TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
            print('Test size: ', TEST_SIZE)
            VAL_SIZE = int(TOTAL_SIZE * 0.1)
            print('Validation size: ', VAL_SIZE)

            print('-----Selecting Small Pieces from the Original Cube Data-----')
            train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
                                                                         TOTAL_SIZE, total_indices, VAL_SIZE,
                                                                         whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION,
                                                                         batch_size, gt)

            tic1 = time.clock()
            train(net, train_iter, valida_iter, loss, optimizer, device, saved, epochs=num_epochs)
            toc1 = time.clock()

            pred_test = []
            #     import pdb;pdb.set_trace()
            tic2 = time.clock()
            with torch.no_grad():
                for X, y in test_iter:
                    X = X.to(device)
                    net.eval()  # 评估模式, 这会关闭dropout
                    y_hat = net(X)
                    # print(net(X))
                    pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
            toc2 = time.clock()
            collections.Counter(pred_test)
            gt_test = gt[test_indices] - 1
            overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
            confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
            each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
            kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

            torch.save(net.state_dict(), saved + "/" + str(round(overall_acc, 3)) + '.pth')
            KAPPA.append(kappa)
            OA.append(overall_acc)
            AA.append(average_acc)
            TRAINING_TIME.append(toc1 - tic1)
            TESTING_TIME.append(toc2 - tic2)
            ELEMENT_ACC[index_iter, :] = each_acc

        print("--------" + net.module.name + " Training Finished-----------")

        record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                      saved + '/' + net.module.name + day_str + '_' + args.dataset + 'split：' + str(TRAIN_SPLIT) + 'lr：' + str(
                          lr) + '.txt')

        generate_png(all_iter, net, gt_hsi, args.dataset, device, total_indices, saved)


if __name__ == "__main__":
    main()
