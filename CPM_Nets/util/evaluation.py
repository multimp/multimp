import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
#tf.enable_eager_execution()
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
#tf.compat.v1.enable_eager_execution()
import torch
import pickle5 as pickle
from util.util import read_data
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    #m = np.mean(data)
    #std = np.std(data)
    return (data - m) / (std + 1e-3)

def transform_format(data_both, idx_record):
    data= dict()
    data_cat = dict()
    for i_view in data_both.keys():
        data[i_view] = data_both[i_view][:, idx_record[i_view]['value']]
        if len(idx_record[i_view]['cat']) > 0:
            for i_cat in idx_record[i_view]['cat']:
                current_cat = data_both[i_view][:, idx_record[i_view]['cat'][i_cat]]
                #current_cat[current_cat == current_cat.max()] = 1
                #current_cat[current_cat < current_cat.max()] = 0
                #ohe = OneHotEncoder()
                #current_cat = ohe.inverse_transform(current_cat)
                current_cat = np.argmax(current_cat, axis=1)
                data[i_view] = np.concatenate((data[i_view], current_cat[:, None]), axis=1)
    return data

def evaluate(original, imputed, sn, original_MX, idx_record_both, cat_indicator, view_num):
    imputed = transform_format(imputed, idx_record_both)
    tf.executing_eagerly()
    mx_ori_nume = dict()
    mx_ori_cat = dict()
    mx_ori = dict()
    sn_for_testing_nume = dict()
    sn_for_testing_cat = dict()
    sn_for_testing = dict()
    error_nume = dict()
    error_cat = dict()
    imputed_cat = dict()
    imputed_nume = dict()
    original_cat = dict()
    original_nume = dict()
    # split values according to numerical and categorical
    for i_view in range(int(view_num)):
        cat_indicator[str(i_view)] = cat_indicator[str(i_view)].astype('bool')
        mx_ori_nume[str(i_view)] = \
            np.logical_not(original_MX[str(i_view)])[:, np.logical_not(cat_indicator[str(i_view)])]
        mx_ori_cat[str(i_view)] = \
            np.logical_not(original_MX[str(i_view)])[:, cat_indicator[str(i_view)]]
        mx_ori[str(i_view)] = \
            np.logical_not(original_MX[str(i_view)])
        sn_for_testing[str(i_view)] = \
            np.logical_xor(np.logical_not(sn[str(i_view)]), mx_ori[str(i_view)]).astype('bool')

        sn_for_testing_cat[str(i_view)] = \
            ((1 - sn[str(i_view)]) - mx_ori[str(i_view)])[:, cat_indicator[str(i_view)]].astype('bool')
        sn_for_testing_nume[str(i_view)] = \
            ((1 - sn[str(i_view)]) - mx_ori[str(i_view)])[:, np.logical_not(cat_indicator[str(i_view)])].astype('bool')
        imputed_cat[str(i_view)] = imputed[str(i_view)][:, cat_indicator[str(i_view)]]
        imputed_nume[str(i_view)] = imputed[str(i_view)][:, np.logical_not(cat_indicator[str(i_view)])]
        original_cat[str(i_view)] = original[str(i_view)][:, cat_indicator[str(i_view)]]
        original_nume[str(i_view)] = original[str(i_view)][:, np.logical_not(cat_indicator[str(i_view)])]

    error_nume[str(i_view)] = []
    for i_view in range(int(view_num)):
        '''
        for ith_col in range(imputed_nume[str(i_view)].shape[1]):
            maxv = original_nume[str(i_view)][:, ith_col][[sn_for_testing_nume[str(i_view)]]].max()
            minv = original_nume[str(i_view)][:, ith_col][[sn_for_testing_nume[str(i_view)]]].min()
            current_mse = ((imputed_nume[str(i_view)][:, ith_col][sn_for_testing_nume[str(i_view)][:, ith_col]] -
                                                       original_nume[str(i_view)][:, ith_col][sn_for_testing_nume[str(i_view)][:, ith_col]])**2).sum()
            error_nume[str(i_view)].append((current_mse**0.5)/(maxv - minv))
        '''

        error_nume[str(i_view)] = \
            ((imputed_nume[str(i_view)][sn_for_testing_nume[str(i_view)]] -
             original_nume[str(i_view)][sn_for_testing_nume[str(i_view)]])**2).sum()
    error_cat = []
    for i_view in range(int(view_num)):
        if cat_indicator[str(i_view)].sum() > 0:
            for ith_col in range(original_cat[str(i_view)].shape[1]):
                #catv = np.unique(original_cat[str(i_view)][:, ith_col])
                #gt = np.zeros_like(original_cat[str(i_view)][:, ith_col])
                #gt[original_cat[str(i_view)][:, ith_col] > catv.min()] = 1
                gt = original_cat[str(i_view)][:, ith_col]
                pred = imputed_cat[str(i_view)][:, ith_col]
                try:
                    error_cat.append(accuracy_score(gt,pred))
                except:
                    print(gt)

    # evaluate original mean imputation
    error_nume_meanimp = {}
    error_nume_meanimp[str(i_view)] = []
    for i_view in range(int(view_num)):
        error_nume_meanimp[str(i_view)] = ((0 -
                                    original_nume[str(i_view)][sn_for_testing_nume[str(i_view)]]) ** 2).sum()

    # arrage cross view results
    num_of_values = 0
    total_error_nume = 0
    totoal_error_nume_meanimp = 0
    for i_view in range(int(view_num)):
        num_of_values += sn_for_testing_nume[str(i_view)].sum()
        total_error_nume += error_nume[str(i_view)]
        totoal_error_nume_meanimp += error_nume_meanimp[str(i_view)]
    mean_error_nume = (total_error_nume / num_of_values)**0.5
    mean_error_nume_meanimp = (totoal_error_nume_meanimp / num_of_values) ** 0.5
    mean_acc = np.array(error_cat).mean()

    print('Original MSE is {:.4f},'.format(mean_error_nume_meanimp))


    return mean_error_nume, mean_acc, imputed, sn_for_testing





def classification_evaluation(feature_path, label_path):
    with open(feature_path, 'rb') as handle:
        data = pickle.load(handle)

    with open(label_path, 'rb') as handle:
        label = pickle.load(handle)

    for i in data.keys():
        if i == '0':
            features = data[i]
        else:
            features = np.concatenate((features, data[i]), axis=1)

    labels = label['gt']

    train_idx, test_idx, labels_train, labels_test = \
        train_test_split(np.arange(len(labels)),
                         labels.ravel(),
                         train_size=0.8,
                         random_state=0,
                         stratify=labels.ravel())

    clf= SVC(kernel='linear')
    clf.fit(features[train_idx], labels[train_idx].ravel())
    accuracy = clf.score(features[test_idx], labels[test_idx].ravel())
    return accuracy


def batch_evaluation(root, results_path, savepath, gtpath):
    import os
    import pandas as pd
    methods = ['CPMNets', 'CPMNets_ori']
    mv = ['multiview_True', 'multiview_False']
    missing_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dict_results = pd.read_csv(results_path, header=0)
    #for imethod in methods:
    #    for i_view in mv:
    #        for i_mr in missing_rate:
    accs = []
    for i in range(len(dict_results)):
        imethod = dict_results['model'][i]
        i_view = dict_results['multi_view'][i]
        i_mr = dict_results['missing_rate'][i]
        path = os.path.join(root, imethod + '_multiview_' + str(i_view), str(i_mr))
        feature_path = os.path.join(path, 'adni_missing_rate_' + str(i_mr) + '.pkl')
        acc = classification_evaluation(feature_path, label_path=gtpath)
        dict_results = pd.read_csv(results_path, header=0)
        accs.append(acc)
    dict_results.insert(dict_results.shape[1],
                              'accuracy',
                              accs)
    dict_results.to_csv(savepath, index=None)
    return dict_results
