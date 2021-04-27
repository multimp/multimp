import numpy as np
from util.util import read_data
from util.get_sn import get_sn
from util.model import CPMNets
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score
import os
import warnings
from util.evaluation import *
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle5 as pickle


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=512,
                        help='dimensionality of the latent space data [default: 512]')
    parser.add_argument('--epochs-train', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 20]')
    parser.add_argument('--epochs-test', type=int, default=1, metavar='N',
                        help='number of epochs to test [default: 50]')
    parser.add_argument('--lamb', type=float, default=10.,
                        help='trade off parameter [default: 10]')
    parser.add_argument('--missing-rate', type=float, default=0.,
                        help='view missing rate [default: 0]')
    parser.add_argument('--log-dir', default='/playpen-raid/data/oct_yining/multimp/results/', type=str,
                        help='saving path')
    parser.add_argument('--unsu', type=bool, default=True,
                        help='view missing rate [default: 0]')

    args = parser.parse_args()

    # read data
    if args.unsu:
        allData, trainData, testData, view_num = read_data('/playpen-raid/data/oct_yining/multimp/data/adni_tabular.pkl', Normal=1)
        # Randomly generated missing matrix
        Sn_all = get_sn(allData, view_num, trainData.num_examples, args.missing_rate)
    else:
        trainData, testData, view_num = read_data('/playpen-raid/data/oct_yining/multimp/data/adni_tabular.pkl',
                                                  ratio=0.8, Normal=1)

    # Randomly generated missing matrix
    Sn_train = get_sn(trainData, view_num, trainData.num_examples, args.missing_rate)
    Sn_test = get_sn(testData, view_num, testData.num_examples, args.missing_rate)
    outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]
    # set layer size
    layer_size = [[outdim_size[i]] for i in range(view_num)]
    layer_size_d = [[outdim_size[i], 128, 1] for i in range(view_num)]
    # set parameter
    epoch = [args.epochs_train, args.epochs_test]
    learning_rate = [0.001, 0.01]


    # train
    if args.unsu:
        # Model building
        model = CPMNets(view_num,
                        allData.cat_indicator,
                        allData.num_examples,
                        testData.num_examples,
                        layer_size, layer_size_d,
                        args.lsd_dim,
                        learning_rate,
                        args.lamb)
        model.train(allData.data, Sn_all, allData.labels.reshape(allData.num_examples, 1), epoch[0])
        #H_all = model.get_h_all()
        # get recovered matrix

        imputed_data = model.recover(allData.data, Sn_all, allData.labels.reshape(allData.num_examples, 1))
        mean_mse, mean_auc = evaluate(allData.data, imputed_data, Sn_all, model.cat_indicator, view_num)
        print('MSE is {:.4f}, MeanAUC is {:.4f}'.format(mean_mse, mean_auc))

        # save results
        root_dir = args.log_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        metrics_path = os.path.join(root_dir, 'metrics')
        mat_path = os.path.join(root_dir, 'imputed')
        if not os.path.exists(metrics_path):
            os.mkdir(metrics_path)
        if not os.path.exists(mat_path):
            os.mkdir(mat_path)
        mat_path = mat_path + '/adni_missing_rate_' + str(args.missing_rate) + '.pkl'
        metrics_path =  metrics_path +  '/results.csv'

        ## caculate results
        current_metrics = {}
        current_metrics['missing_rate'] = args.missing_rate
        current_metrics['mse'] = mean_mse
        current_metrics['auc'] = mean_auc
        current_metrics['epoch'] = mean_auc
        ## save to pickle
        with open(mat_path, 'wb') as handle:
            pickle.dump(imputed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ## save to csv
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path, header=0)
            metrics.append(pd.DataFrame(current_metrics, columns=['missing_rate', 'auc', 'mse'], index=[len(metrics)]))
        else:
            pd.DataFrame(current_metrics, index=[0], columns=['missing_rate', 'auc', 'mse', 'epochs']).to_csv(metrics_path)

        # test
        if args.unsu:
            model.test(testData.data, Sn_test, testData.labels.reshape(testData.num_examples, 1), epoch[1])
        H_test = model.get_h_test()
        #label_pre = classfiy.ave(H_train, H_test, trainData.labels)
        #print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre)))







    # supervised
    else:
        model = CPMNets(view_num,
                        trainData.cat_indicator,
                        trainData.num_examples,
                        testData.num_examples,
                        layer_size, layer_size_d,
                        args.lsd_dim,
                        learning_rate,
                        args.lamb)
        model.train(trainData.data, Sn_train, trainData.labels.reshape(trainData.num_examples, 1), epoch[0])
        H_train = model.get_h_train()

    # get recovered matrix
    imputed_data = dict()
    for v_num in range(model.view_num):
        imputed_data[str(v_num)] = model.Encoding_net(H_train, v_num)
    mean_mse, mean_auc = evaluate(trainData.data, imputed_data, Sn_train, model.cat_indicator, view_num)
    print('MSE is {:.4f}, MeanAUC is {:.4f}'.format(accuracy_score(mean_mse, mean_auc)))

    # save results

    # test
    if args.unsu:

        model.test(testData.data, Sn_test, testData.labels.reshape(testData.num_examples, 1), epoch[1])
    H_test = model.get_h_test()
    label_pre = classfiy.ave(H_train, H_test, trainData.labels)
    print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre)))