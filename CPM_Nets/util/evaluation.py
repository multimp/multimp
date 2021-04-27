import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
#tf.enable_eager_execution()
from sklearn.metrics import roc_auc_score
#tf.compat.v1.enable_eager_execution()
import torch
def evaluate(original, imputed, sn, original_MX, cat_indicator, view_num):
    tf.executing_eagerly()
    mx_ori_nume = dict()
    mx_ori_cat = dict()
    mx_ori = dict()
    sn_for_testing_nume = dict()
    sn_for_testing_cat = dict()
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
        sn_for_testing_cat[str(i_view)] = \
            ((1 - sn[str(i_view)]) - mx_ori[str(i_view)])[:, cat_indicator[str(i_view)]].astype('bool')
        sn_for_testing_nume[str(i_view)] = \
            ((1 - sn[str(i_view)]) - mx_ori[str(i_view)])[:, np.logical_not(cat_indicator[str(i_view)])].astype('bool')
        imputed_cat[str(i_view)] = imputed[str(i_view)][:, cat_indicator[str(i_view)]]
        imputed_nume[str(i_view)] = imputed[str(i_view)][:, np.logical_not(cat_indicator[str(i_view)])]
        original_cat[str(i_view)] = original[str(i_view)][:, cat_indicator[str(i_view)]]
        original_nume[str(i_view)] = original[str(i_view)][:, np.logical_not(cat_indicator[str(i_view)])]

    for i_view in range(int(view_num)):
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
    # arrage cross view results
    num_of_values = 0
    total_error_nume = 0
    for i_view in range(int(view_num)):
        num_of_values += sn_for_testing_nume[str(i_view)].sum()
        total_error_nume += error_nume[str(i_view)]
    mean_error_nume = (total_error_nume / num_of_values)**0.5
    mean_auc = np.array(error_cat).mean()
    return mean_error_nume, mean_auc