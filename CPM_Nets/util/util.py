import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle
import tensorflow as tf
import pickle5 as pickle
from sklearn.model_selection import train_test_split

from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#np.random.seed(0)
from sklearn.preprocessing import MinMaxScaler

class DataSet(object):

    def __init__(self, data, view_number, labels, cat_indicator=None, multi_view=True, missing_rate=0.1):
        """
        Construct a DataSet.
        """
        self.missing_rate = missing_rate
        self.multi_view = multi_view
        self.data = dict()
        self.MX = dict()
        self._num_examples = data[0].shape[0]
        self._labels = labels

        # processing multi_view
        if multi_view:
            '''
            if cat_indicator is None:
                self.cat_indicator = dict()
                for iv in range(int(view_number)):
                    self.cat_indicator[str(iv)] = np.zeros(data[iv].shape[1]).astype('int')
            else:
                self.cat_indicator = cat_indicator
            '''

            self.cat_indicator = cat_indicator

            for v_num in range(view_number):
                self.data[str(v_num)] = data[v_num]
            for v_num in range(view_number):
                self.MX[str(v_num)] = np.logical_not(np.isnan(data[v_num]))
                if np.isnan(self.data[str(v_num)]).sum() > 0:
                    for ith_col in range(self.data[str(v_num)].shape[1]):
                        imputed = self.data[str(v_num)][:, ith_col][self.MX[str(v_num)][:, ith_col]].mean()
                        self.data[str(v_num)][:, ith_col] = np.nan_to_num(self.data[str(v_num)][:, ith_col], nan=imputed)#imputed)
                #self.data[str(v_num)] = np.nan_to_num(self.data[str(v_num)], nan=0)

            # generate missing values
            self.Sn = self.get_sn()
            self.data_both, self.Sn_both, self.idx_record_both, self.data, self.Sn, self.MX = \
                self.transform_to_binaries(self.data, self.Sn, self.cat_indicator, self.MX)

            for v_num in self.data_both.keys():
                self.data_both[str(v_num)][:, self.idx_record_both[str(v_num)]['value']] = \
                    Normalize(self.data_both[str(v_num)][:, self.idx_record_both[str(v_num)]['value']])
                self.data_both[str(v_num)] = \
                    self.data_both[str(v_num)] * self.Sn_both[str(v_num)].astype('float')
                self.data[str(v_num)][:, np.logical_not(self.cat_indicator[str(v_num)])] = \
                    Normalize(self.data[str(v_num)][:, np.logical_not(self.cat_indicator[str(v_num)])])

        # merging multi-view data to one view
        else:

            self.cat_indicator = {}
            for iv in range(int(view_number)):
                if iv == 0:
                    self.cat_indicator['0'] = cat_indicator['0'].astype('bool')
                else:
                    self.cat_indicator['0'] = \
                    np.concatenate((self.cat_indicator['0'],
                                    cat_indicator[str(iv)].astype('bool')), axis=0)

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
                    self.data['0'] = data[v_num]
                    self.data['0'][:, np.logical_not(cat_indicator[str(v_num)])] = \
                        Normalize(data[0][:, np.logical_not(cat_indicator[str(v_num)])])
                else:
                    current_view = data[v_num].copy()
                    current_view[:, np.logical_not(self.cat_indicator[str(v_num)])] = \
                        Normalize(data[v_num][:, np.logical_not(self.cat_indicator[str(v_num)])])
                    self.data['0'] = \
                        np.concatenate((self.data['0'], current_view), axis=1)

            # generate missing values
            self.Sn = self.get_sn()
            # self.value_data, \
            # self.cat_data, \
            # self.value_Sn, \
            # self.cat_MX, \
            # self.record_view_size = \
            self.data_both, self.Sn_both, self.idx_record_both, self.data, self.Sn, self.MX = \
                self.transform_to_binaries(self.data, self.Sn, self.cat_indicator, self.MX)
            '''
            for v_num in self.data_both.keys():
                self.data_both[str(v_num)][:, self.idx_record_both[str(v_num)]['value']] = \
                    Normalize(self.data_both[str(v_num)][:, self.idx_record_both[str(v_num)]['value']])
                self.data_both[str(v_num)] = \
                    self.data_both[str(v_num)] * self.Sn_both[str(v_num)].astype('float')
                self.data[str(v_num)][:, np.logical_not(self.cat_indicator[str(v_num)])] = \
                    Normalize(self.data[str(v_num)][:, np.logical_not(self.cat_indicator[str(v_num)])])
                    
            '''

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


    def get_sn(self):
        """Randomly generate incomplete data information, simulate partial view data with complete view data
        :param view_num:view number
        :param alldata_len:number of samples
        :param missing_rate:Defined in section 3.2 of the paper
        :return:Sn
        """

        # generating some missing values other than current missing
        ratio = 1 - self.missing_rate
        MX = self.MX.copy()
        current_MX = dict()
        for i_view in MX.keys():
            number_of_observed = MX[i_view].sum()
            matrix_iter = (randint(0, 100, size=number_of_observed, ) < int(ratio * 100)).astype(np.int)
            a = MX[i_view].copy()
            a[MX[i_view]] = matrix_iter
            current_MX[i_view] = a  # [MX[i_view]] = matrix_iter
        return current_MX

    def transform_to_binaries(self, data, Sn, cat_indicator, MX):
        cat_data_multi_cls = dict()
        value_data = dict()
        cat_data = dict()
        value_Sn = dict()
        cat_Sn_multi_cls = dict()
        cat_Sn = dict()
        #MX = dict()
        value_MX = dict()
        cat_MX = dict()

        for i_view in data.keys():
            if cat_indicator[i_view].sum() == 0:
                cat_data_multi_cls[i_view] = []
                value_data[i_view] = data[i_view]
                value_Sn[i_view] = Sn[i_view]
                cat_Sn_multi_cls[i_view] = []
                value_MX[i_view] = MX[i_view]
                cat_MX[i_view] = []
            else:
                cat_data_multi_cls[i_view] = data[i_view][:, cat_indicator[i_view]]
                value_data[i_view] = data[i_view][:, np.logical_not(cat_indicator[i_view])]
                cat_Sn_multi_cls[i_view] = Sn[i_view][:, cat_indicator[i_view]]
                value_Sn[i_view] = Sn[i_view][:, np.logical_not(cat_indicator[i_view])]
                value_MX[i_view] = MX[i_view][:, np.logical_not(cat_indicator[i_view])]
                cat_MX[i_view] = MX[i_view][:, cat_indicator[i_view]]


        record_cat_size = dict()
        record_view_size = dict()

        for i_view in cat_data_multi_cls.keys():
            cat_data[i_view] = dict()
            cat_Sn[i_view] = dict()
            #record_cat_size[i_view] = 0
            #record_view_size[i_view] = value_data[i_view].shape[1]
            if len(cat_data_multi_cls[i_view]) > 0:
                for i_cat_feat in range(cat_data_multi_cls[i_view].shape[1]):
                    current_cat = cat_data_multi_cls[i_view][:, i_cat_feat].astype('int')
                    encoder_onehot = OneHotEncoder()
                    encoded_labels = encoder_onehot.fit_transform(current_cat[:, None]).toarray()
                    cat_data[i_view][i_cat_feat] = encoded_labels
                    #self.record_cat_size[i_view] += encoded_labels.shape[1]
                    #self.record_view_size[i_view] += self.record_cat_size[i_view]
                    cat_Sn[i_view][i_cat_feat] = \
                        np.tile(cat_Sn_multi_cls[i_view][:, i_cat_feat][:, None], (1, encoded_labels.shape[1]))
            else:
                cat_data[i_view] =[]
                cat_Sn[i_view] = []

        data_both = dict()
        Sn_both = dict()
        idx_record = dict()
        data_ori = dict()
        Sn_ori = dict()
        MX_ori = dict()

        for i_view in data.keys():
            data_both[i_view] = value_data[i_view]
            Sn_both[i_view] = value_Sn[i_view]
            data_ori[i_view] = value_data[i_view]
            Sn_ori[i_view] = value_Sn[i_view]
            idx_record[i_view] = dict()
            MX_ori[i_view] = value_MX[i_view]
            if len(cat_data[i_view]) > 0:
                idx_record[i_view]['cat'] = dict()
                for i_cat in cat_data[i_view].keys():
                    idx_record[i_view]['cat'][i_cat] = \
                        np.arange(cat_Sn[i_view][i_cat].shape[1]) + data_both[i_view].shape[1]
                    data_both[i_view] = np.concatenate((data_both[i_view],
                                                        cat_data[i_view][i_cat]),
                                                            axis=1)
                    Sn_both[i_view] = np.concatenate((Sn_both[i_view],
                                                      cat_Sn[i_view][i_cat]),
                                                            axis=1)
                    data_ori[i_view] = np.concatenate((data_ori[i_view],
                                                        cat_data_multi_cls[i_view][:, [i_cat]]),
                                                       axis=1)
                    Sn_ori[i_view] = np.concatenate((Sn_ori[i_view],
                                                      cat_Sn_multi_cls[i_view][:, [i_cat]]),
                                                     axis=1)
                    MX_ori[i_view] = np.concatenate((MX_ori[i_view],
                                                      cat_MX[i_view][:, [i_cat]]),
                                                     axis=1)
            else:
                idx_record[i_view]['cat'] = []
            idx_record[i_view]['value'] = np.arange(value_Sn[i_view].shape[1])
            current_cat_indicator = np.ones(self.cat_indicator[i_view].shape)
            current_cat_indicator[idx_record[i_view]['value']] = 0
            self.cat_indicator[i_view] = current_cat_indicator.astype('bool')
        return data_both, Sn_both, idx_record, data_ori, Sn_ori, MX_ori




def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """

    #m = np.mean(data, axis=0)
    #std = np.std(data, axis=0)
    m = np.mean(data)
    std = np.std(data)

    return (data - m) / (std + 1e-3)

    #return data

    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #data = scaler.fit_transform(data)
    #return data


def read_data(str_name, ratio=None, Normal=1, multi_view=True, missing_rate=0):
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
        cat_indicator[str(i)] = np.array(data['CatogoricalIndicator'][i]).astype(dtype='bool')
    # train test spilt
    X_train = []
    X_test = []
    X_all = []
    #if min(data['gt']) == 0:
    #    labels = data['gt'] + 1
    #else:
    #    labels = data['gt']
    #labels = labels.squeeze()
    labels = data['gt']
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
        alldata = DataSet(X_all, view_number, np.array(labels), cat_indicator, multi_view=multi_view, missing_rate=missing_rate)
        traindata = DataSet(X_train, view_number, np.array(labels_train), cat_indicator, multi_view=multi_view, missing_rate=missing_rate)
        testdata = DataSet(X_test, view_number, np.array(labels_test), cat_indicator, multi_view=multi_view, missing_rate=missing_rate)
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
    imputed = dict()
    for ith_view in sn.keys():
        #missingness = np.logical_not(sn[ith_view])
        #originaldata[ith_view][missingness] = imputation[ith_view][missingness]
        #imputation[ith_view][sn[ith_view]] = originaldata[ith_view][sn[ith_view]]
        imputed[ith_view] = originaldata[ith_view] * sn[ith_view].astype('float') + \
                            imputation[ith_view] * np.logical_not(sn[ith_view]).astype('float')
    return imputed