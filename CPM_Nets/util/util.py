import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle
import tensorflow as tf
import pickle5 as pickle
from sklearn.model_selection import train_test_split

class DataSet(object):

    def __init__(self, data, view_number, labels, cat_indicator=None, multi_view=True):
        """
        Construct a DataSet.
        """
        self.multi_view = multi_view
        if multi_view:
            if cat_indicator is None:
                self.cat_indicator = dict()
                for iv in range(int(view_number)):
                    self.cat_indicator[str(iv)] = np.zeros(data[iv].shape[1]).astype('int')
            else:
                self.cat_indicator = cat_indicator

            self.cat_indicator = cat_indicator
            self.data = dict()
            self.MX = dict()
            self._num_examples = data[0].shape[0]
            self._labels = labels
            for v_num in range(view_number):
                self.data[str(v_num)] = data[v_num]

            for v_num in range(view_number):
                self.MX[str(v_num)] = np.logical_not(np.isnan(data[v_num]))
                if np.isnan(self.data[str(v_num)]).sum() > 0:
                    for ith_col in range(self.data[str(v_num)].shape[1]):
                        imputed = self.data[str(v_num)][:, ith_col][np.logical_not(np.isnan(self.data[str(v_num)])[:, ith_col])].mean()
                        self.data[str(v_num)][:, ith_col] = np.nan_to_num(self.data[str(v_num)][:, ith_col], nan=imputed)
                #self.data[str(v_num)] = np.nan_to_num(self.data[str(v_num)], nan=0)
            for v_num in self.data.keys():
                self.data[str(v_num)] = Normalize(self.data[str(v_num)])

        else:

            self.cat_indicator = {}
            for iv in range(int(view_number)):
                if iv == 0:
                    self.cat_indicator['0'] = cat_indicator['0']
                else:
                    self.cat_indicator['0'] = \
                    np.concatenate((self.cat_indicator['0'],
                                    cat_indicator[str(iv)].astype('int')), axis=0)

            self.data = dict()
            self.MX = dict()
            self._num_examples = data[0].shape[0]
            self._labels = labels
            #for v_num in range(view_number):
            #    self.data[str(v_num)] = data[v_num]

            for v_num in range(view_number):
                if v_num == 0:
                    self.MX['0'] = np.logical_not(np.isnan(data[v_num]))
                else:
                    self.MX['0'] = np.concatenate((self.MX['0'],
                                                   np.logical_not(np.isnan(data[v_num]))), axis=1)
                if np.isnan(data[v_num]).sum() > 0:
                    for ith_col in range(data[v_num].shape[1]):
                        imputed = data[v_num][:, ith_col][
                            np.logical_not(np.isnan(data[v_num])[:, ith_col])].mean()
                        data[v_num][:, ith_col] = np.nan_to_num(data[v_num][:, ith_col], nan=imputed)
                # self.data[str(v_num)] = np.nan_to_num(self.data[str(v_num)], nan=0)
            for v_num in range(len(data)):
                if v_num == 0:
                    self.data['0'] = Normalize(data[0])
                else:
                    self.data['0'] = \
                        np.concatenate((self.data['0'], Normalize(data[v_num])), axis=1)


    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

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


def read_data(str_name, ratio=None, Normal=1, multi_view=True):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    with open(str_name, 'rb') as handle:
        data = pickle.load(handle)
    view_number = len(data['X'])
    X = dict()
    cat_indicator = dict()
    for i in range(view_number):
        X[i] = []
        for ith_sub in range(data['X'].shape[1]):
            X[i].append(np.array(data['X'][i][ith_sub], dtype=float))
        X[i] = np.array(X[i])
        cat_indicator[str(i)] = np.array(data['CatogoricalIndicator'][i]).astype(dtype='int')
    # train test spilt
    X_train = []
    X_test = []
    X_all = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    labels = labels.squeeze()

    if ratio is None:
        train_idx, test_idx, labels_train,  labels_test = \
            train_test_split(np.arange(len(labels)),
                             labels,
                             train_size=ratio,
                             random_state=0,
                             stratify=labels)

        for v_num in range(view_number):
            X_train.append(X[v_num][train_idx])
            X_test.append(X[v_num][test_idx])
            X_all.append(X[v_num])

        '''
        if (Normal == 1):
            for v_num in range(view_number):
                X_train[v_num] = Normalize(X_train[v_num])
                X_test[v_num] = Normalize(X_test[v_num])
        '''
        alldata = DataSet(X_all, view_number, np.array(labels), cat_indicator, multi_view=multi_view)
        traindata = DataSet(X_train, view_number, np.array(labels_train), cat_indicator, multi_view=multi_view)
        testdata = DataSet(X_test, view_number, np.array(labels_test), cat_indicator, multi_view=multi_view)
        if multi_view:
            return alldata, traindata, testdata, view_number
        else:
            return alldata, traindata, testdata, 1
    else:
        train_idx, test_idx, labels_train,  labels_test = \
            train_test_split(np.arange(len(labels)),
                             labels,
                             train_size=ratio,
                             random_state=0,
                             stratify=labels)

        for v_num in range(view_number):
            X_train.append(X[v_num][train_idx])
            X_test.append(X[v_num][test_idx])

        '''
        if (Normal == 1):
            for v_num in range(view_number):
                X_train[v_num] = Normalize(X_train[v_num])
                X_test[v_num] = Normalize(X_test[v_num])
        '''

        traindata = DataSet(X_train, view_number, np.array(labels_train), cat_indicator, multi_view)
        testdata = DataSet(X_test, view_number, np.array(labels_test), cat_indicator, multi_view)
        if multi_view:
            return traindata, testdata, view_number
        else:
            return traindata, testdata, 1



def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def impute_missing_values_using_imputed_matrix(originaldata, imputation, sn):
    for ith_view in sn.keys():
        missingness = np.logical_not(sn[ith_view])
        originaldata[ith_view][missingness] = imputation[ith_view][missingness]
    return originaldata