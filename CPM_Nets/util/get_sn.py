import numpy as np
from numpy.random import randint


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

    return current_MX


def save_Sn(Sn, str_name):
    np.savetxt(str_name + '.csv', Sn, delimiter=',')


def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')
