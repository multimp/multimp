import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_sn(Data, view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """

    # generating some missing values other than current missing
    ratio = 1 - missing_rate
    MX = Data.MX.copy()
    current_MX = dict()
    for i_view in MX.keys():
        number_of_observed = MX[i_view].sum()
        matrix_iter = (randint(0, 100, size=number_of_observed) < int(ratio*100)).astype(np.int)
        a = MX[i_view].copy()
        a[MX[i_view]] = matrix_iter
        current_MX[i_view] = a #[MX[i_view]] = matrix_iter

    '''
    one_rate = 1-missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    '''
    return current_MX


def save_Sn(Sn, str_name):
    np.savetxt(str_name + '.csv', Sn, delimiter=',')


def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')
