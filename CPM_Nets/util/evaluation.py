import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
#tf.enable_eager_execution()
from sklearn.metrics import roc_auc_score
#tf.compat.v1.enable_eager_execution()
import torch
import pickle5 as pickle
from util.util import read_data
from sklearn.svm import SVC
def evaluate(original, imputed, sn, original_MX, cat_indicator, view_num):
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
        error_nume[str(i_view)] = ((imputed_nume[str(i_view)][sn_for_testing_nume[str(i_view)]] -
                                                 original_nume[str(i_view)][sn_for_testing_nume[str(i_view)]])**2).sum()
    error_cat = []
    for i_view in range(int(view_num)):
        if original_cat[str(i_view)].sum() > 0:
            for ith_col in range(original_cat[str(i_view)].shape[1]):
                catv = np.unique(original_cat[str(i_view)][:, ith_col])
                gt = np.zeros_like(original_cat[str(i_view)][:, ith_col])
                gt[original_cat[str(i_view)][:, ith_col] > catv.min()] = 1
                pred = torch.sigmoid(torch.from_numpy(imputed_cat[str(i_view)][:, ith_col])).numpy()
                try:
                    error_cat.append(roc_auc_score(gt,pred))
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
    mean_auc = np.array(error_cat).mean()

    print('Original MSE is {:.4f},'.format(mean_error_nume_meanimp))


    return mean_error_nume, mean_auc, sn_for_testing





def classification_evaluation(feature_path, gt):
    with open(feature_path, 'rb') as handle:
        data = pickle.load(handle)
    trainData, testData, view_num = \
        read_data(feature_path,  ratio=0.8, Normal=1, multi_view = True)


    for i in trainData.keys():
        if i == '0':
            train_features =  trainData['0']
        else:
            train_features = np.concatenate((train_features,))

    clf= SVC(kernel='linear')
    clf.fit(train_features, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))]
    for
