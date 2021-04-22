import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle
import tensorflow as tf
import pickle5 as pickle
from sklearn.model_selection import train_test_split

class DataSet(object):

    def __init__(self, data, view_number, labels, idx):
        """
        Construct a DataSet.
        """
        self.idx = idx
        self.data = dict()
        self.MX = dict()
        self._num_examples = data[0].shape[0]
        self._labels = labels
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

        for v_num in range(view_number):
            self.MX[str(v_num)] = np.logical_not(np.isnan(data[v_num]))
            for ith_col in range(self.data[str(v_num)].shape[1]):
                imputed = np.isreal(self.data[str(v_num)][:, ith_col]).mean()
                self.data[str(v_num)][:, ith_col] = np.nan_to_num(data[v_num][:, ith_col], nan=imputed)
        for v_num in self.data.keys():
            self.data[v_num] = Normalize(self.data[v_num])



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
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def read_data(str_name, ratio, Normal=1):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    if str_name[-3::] == 'mat':
        data = sio.loadmat(str_name)
        view_number = data['X'].shape[1]
        X = np.split(data['X'], view_number, axis=1)
        for i in range(len(X)):
            X[i] = X[i][0][0].T
        # train test spilt
        X_train = []
        X_test = []
        labels_train = []
        labels_test = []
        if min(data['gt']) == 0:
            labels = data['gt'] + 1
        else:
            labels = data['gt']
        classes = max(labels)[0]
        all_length = 0
        for c_num in range(1, classes + 1):
            c_length = np.sum(labels == c_num)
            index = np.arange(c_length)
            shuffle(index)
            labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio)])
            labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
            X_train_temp = []
            X_test_temp = []
            for v_num in range(view_number):
                X_train_temp.append(X[v_num][all_length + index][0:math.floor(c_length * ratio)])
                X_test_temp.append(X[v_num][all_length + index][math.floor(c_length * ratio):])
            if c_num == 1:
                X_train = X_train_temp
                X_test = X_test_temp
            else:
                for v_num in range(view_number):
                    X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                    X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
            all_length = all_length + c_length
        if (Normal == 1):
            for v_num in range(view_number):
                X_train[v_num] = Normalize(X_train[v_num])
                X_test[v_num] = Normalize(X_test[v_num])

        traindata = DataSet(X_train, view_number, np.array(labels_train), )
        testdata = DataSet(X_test, view_number, np.array(labels_test))
        return traindata, testdata, view_number

    else:
        with open(str_name, 'rb') as handle:
            data = pickle.load(handle)
        view_number = len(data['X'])
        X = dict()

        for i in range(view_number):
            X[i] = []
            for ith_sub in range(data['X'].shape[1]):
                X[i].append(np.array(data['X'][i][ith_sub], dtype=float))
            X[i] = np.array(X[i])
        # train test spilt
        X_train = []
        X_test = []
        if min(data['gt']) == 0:
            labels = data['gt'] + 1
        else:
            labels = data['gt']
        labels = labels.squeeze()
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

        traindata = DataSet(X_train, view_number, np.array(labels_train), train_idx)
        testdata = DataSet(X_test, view_number, np.array(labels_test), test_idx)
        return traindata, testdata, view_number




def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

