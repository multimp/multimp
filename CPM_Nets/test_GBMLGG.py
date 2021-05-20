# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 00:47:44 2021

@author: wancenmu
"""

import numpy as np
from util.util import read_data, impute_missing_values_using_imputed_matrix
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
from util.model_ori import CPMNets_ori
from util.model_nume import CPMNets_num
from util.model_nume_ori import CPMNets_num_ori
warnings.filterwarnings("ignore")

#np.random.seed(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CPMNets_num',
                        help='CPMNets, CPMNets_ori')
    parser.add_argument('--lsd-dim', type=int, default=128,
                        help='dimensionality of the latent space data [default: 512]')
    parser.add_argument('--epochs-train', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 20]')
    parser.add_argument('--epochs-test', type=int, default=1, metavar='N',
                        help='number of epochs to test [default: 50]')
    parser.add_argument('--lamb', type=float, default=10.,
                        help='trade off parameter [default: 10]')
    parser.add_argument('--missing-rate', type=float, default=0.1,
                        help='view missing rate [default: 0]')
    parser.add_argument('--log-dir', default='/playpen-ssd/wancen/multimp/results', type=str,
                        help='saving path')
    parser.add_argument('--unsu', type=bool, default=True,
                        help='view missing rate [default: 0]')
    parser.add_argument('--multi-view', type=int, default=0,
                        help='whether to use multiview learning')
    parser.add_argument('--run-idx', type=int, default=0,
                        help='number of run')


    args = parser.parse_args()
    MULTI_VIEW = bool(args.multi_view)
    print('We are training ' + args.model + ', missing rate is ' + str(args.missing_rate) + ' for multiview ' + str(MULTI_VIEW) + '.')
    # read data
    if args.unsu:
        allData, trainData, testData, view_num = read_data('/home/wancen/data/TCGA-GBM_cate2.pkl',
                                                           Normal=1,
                                                           multi_view=MULTI_VIEW,
                                                           missing_rate=args.missing_rate)
        # Randomly generated missing matrix

    else:
        trainData, testData, view_num = read_data('/home/wancen/data/TCGA-GBM_cate2.pkl',
                                                  ratio=0.8, Normal=1, multi_view=MULTI_VIEW,
                                                  missing_rate=args.missing_rate)

    '''
    # Randomly generated missing matrix
    Sn_train = get_sn(trainData, view_num, trainData.num_examples, args.missing_rate)
    Sn_test = get_sn(testData, view_num, testData.num_examples, args.missing_rate)
    '''
    # Randomly generated missing matrix
    Sn_train = allData.Sn_both
    outdim_size = [allData.data_both[str(i)].shape[1] for i in range(view_num)]

    # set layer size
    layer_size = [[outdim_size[i]] for i in range(view_num)]
    layer_size_d = [[np.logical_not(allData.cat_indicator[str(i)]).sum(), 1] for i in range(view_num)]
    # set parameter
    epoch = [args.epochs_train, args.epochs_test]
    learning_rate = [0.001, 0.01]


    # train
    if args.unsu:
        if args.model == 'CPMNets':
            # Model building
            model = CPMNets(view_num,
                            allData.idx_record_both,
                            allData.num_examples,
                            testData.num_examples,
                            layer_size, layer_size_d,
                            args.lsd_dim,
                            learning_rate,
                            args.lamb)
            a = allData.Sn_both.copy()
            b = allData.data_both.copy()
            model.train(b, a, allData.labels.copy(), epoch[0])
        elif args.model == 'CPMNets_ori':
            model = CPMNets_ori(view_num,
                            allData.idx_record_both,
                            allData.num_examples,
                            testData.num_examples,
                            layer_size,
                            args.lsd_dim,
                            learning_rate,
                            args.lamb)
            model.train(allData.data_both.copy(), allData.Sn_both.copy(), allData.labels.copy(), epoch[0])
        elif args.model == 'CPMNets_num_ori':
            model = CPMNets_num_ori(view_num,
                                allData.idx_record_both,
                                allData.num_examples,
                                testData.num_examples,
                                layer_size,
                                args.lsd_dim,
                                learning_rate,
                                args.lamb)
            model.train(allData.data_both.copy(), allData.Sn_both.copy(), allData.labels.copy(), epoch[0])
        elif args.model == 'CPMNets_num':
            model = CPMNets_num(view_num,
                            allData.idx_record_both,
                            allData.num_examples,
                            testData.num_examples,
                            layer_size, layer_size_d,
                            args.lsd_dim,
                            learning_rate,
                            args.lamb)
            a = allData.Sn_both.copy()
            b = allData.data_both.copy()
            model.train(b, a, allData.labels.copy(), epoch[0])

        #H_all = model.get_h_all()
        # get recovered matrix

        imputed_data = model.recover(allData.data_both.copy(), allData.Sn_both.copy(), allData.labels.copy())
        imputed_data = transform_format(imputed_data, allData.idx_record_both)
        imputed_data = impute_missing_values_using_imputed_matrix(allData.data.copy(), imputed_data.copy(), allData.Sn)
        latent_vectors = model.get_h_all()
        # evaluete method
        mean_mse, mean_acc, added_missingness = \
            evaluate(allData.data,
                     imputed_data,
                     allData.Sn,
                     allData.MX,
                     allData.idx_record_both,
                     allData.cat_indicator,
                     view_num)
        print('MSE is {:.4f}, MeanACC is {:.4f}'.format(mean_mse, mean_acc))
        '''
        mean_mse_ini, mean_acc_ini, imputed_data_ini, added_missingness_ini = \
            evaluate(allData.data,
                     allData.data_both,
                     allData.Sn,
                     allData.MX,
                     allData.idx_record_both,
                     allData.cat_indicator,
                     view_num)
        print('Initial MSE is {:.4f}, Initial MeanACC is {:.4f}'.format(mean_mse_ini, mean_acc_ini))
        '''
        # save results
        root_dir = args.log_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        metrics_path = os.path.join(root_dir, 'metrics')
        mat_path = os.path.join(root_dir, 'imputed', str(args.run_idx), args.model + '_multiview_' + str(MULTI_VIEW), str(args.missing_rate))

        print('saving in ' + mat_path)
        if not os.path.exists(os.path.join(root_dir, 'imputed')):
            os.mkdir(os.path.join(root_dir, 'imputed'))
        if not os.path.exists(os.path.join(root_dir, 'imputed', str(args.run_idx))):
            os.mkdir(os.path.join(root_dir, 'imputed', str(args.run_idx)))
        if not os.path.exists(os.path.join(root_dir, 'imputed', str(args.run_idx), args.model + '_multiview_' + str(MULTI_VIEW))):
            os.mkdir(os.path.join(root_dir, 'imputed', str(args.run_idx), args.model + '_multiview_' + str(MULTI_VIEW)))
        if not os.path.exists(os.path.join(root_dir, 'imputed', str(args.run_idx), args.model + '_multiview_' + str(MULTI_VIEW), str(args.missing_rate))):
            os.mkdir(os.path.join(root_dir, 'imputed', str(args.run_idx), args.model + '_multiview_' + str(MULTI_VIEW), str(args.missing_rate)))

        if not os.path.exists(metrics_path):
            os.mkdir(metrics_path)
        mat_file = mat_path + '/GBMLGG_missing_rate_' + str(args.missing_rate) + '.pkl'
        mask_file = mat_path + '/mask_GBMLGG_missing_rate_' + str(args.missing_rate) + '.pkl'
        metrics_file = metrics_path + '/results.csv'
        latent_vectors_file = mat_path + '/latent_GBMLGG_missing_rate_' + str(args.missing_rate) + '.csv'
        labels_file = mat_path + '/labels_GBMLGG_missing_rate_' + str(args.missing_rate) + '.csv'

        ## caculate results
        current_metrics = {}
        current_metrics['missing_rate'] = [args.missing_rate]
        current_metrics['mse'] = [mean_mse]
        current_metrics['acc'] = [mean_acc]
        current_metrics['epoch'] = [int(args.epochs_train)]
        current_metrics['model'] = [args.model]
        current_metrics['multi_view'] = [MULTI_VIEW]
        current_metrics['run_idx'] = [str(args.run_idx)]

        ## save to imputations
        #imputed_data = impute_missing_values_using_imputed_matrix(allData.data.copy(), imputed_data, allData.Sn)
        #with open(mat_file, 'wb') as handle:
            #pickle.dump(imputed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #with open(mask_file, 'wb') as handle:
            #pickle.dump(added_missingness, handle, protocol=pickle.HIGHEST_PROTOCOL)

        pd.DataFrame(latent_vectors).to_csv(latent_vectors_file, index=None, columns=None, header=None)
        pd.DataFrame(allData.labels).to_csv(labels_file, index=None, columns=None, header=None)
        for ith_view in range(int(view_num)):
            mat_path_v = mat_path + '/imputed_GBMLGG_missing_rate_' + str(args.missing_rate) + '_view_' + str(ith_view) + '.csv'
            pd.DataFrame(imputed_data[str(ith_view)]).to_csv(mat_path_v, index=None, columns=None, header=None)
            #mask_path_v = mat_path + '/mask_GBMLGG_missing_rate_' + str(args.missing_rate) + '_view_' + str(ith_view) + '.csv'
            #pd.DataFrame(added_missingness[str(ith_view)]).to_csv(mask_path_v, index=None, columns=None, header=None)

        ## save to csv
        if os.path.exists(metrics_file):
            metrics = pd.read_csv(metrics_file, header=0)
            new_metrics = metrics.append(pd.DataFrame(current_metrics,
                                                      columns=['missing_rate', 'acc', 'mse', 'epoch', 'model', 'multi_view', 'run_idx'], index=[0]))
            new_metrics.to_csv(metrics_file, index=None)
        else:
            pd.DataFrame.from_dict(current_metrics).to_csv(metrics_file, index=None)
