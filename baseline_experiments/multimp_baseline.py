import os
import glob
import sys
import shutil
import json
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from math import sqrt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from datawig import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats
import operator

from fancyimpute_adjusted import (
    MatrixFactorization,
    KNN,
    SimpleFill
)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
DIR_PATH = '.'
dir_path = "."

# this appears to be neccessary for not running into too many open files errors
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))

def dict_product(hp_dict):
    '''
    Returns cartesian product of hyperparameters
    '''
    return [dict(zip(hp_dict.keys(),vals)) for vals in \
            itertools.product(*hp_dict.values())]

def evaluate_mse(X_imputed, X):
    return ((X_imputed - X) ** 2).mean()

def get_name(xx):
    for objname, oid in globals().items():
        if oid is xx:
            return objname

def get_data(path):
    pickle_file = open(path, "rb")
    objects = []
    while True:
        try:
            objects.append(pickle.load(pickle_file))
        except EOFError:
            break
    pickle_file.close()
    return objects

def formatter(x):
    if 0<x<1:
        return round(x)
    else:
        return x
    
def Normalize(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    return data

def get_sn(Data, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    ratio = 1 - missing_rate
    n_features = len(Data[0])
    n_samples = Data.size
    n_elements = n_features * n_samples
    # generating some missing values other than current missing
    MX = (np.random.randint(0, 100, size=n_elements) < int(ratio * 100))
    import operator
    MX = np.array(list(map(operator.not_, MX)))
    MX = np.split(MX, n_samples)

    return MX

X = get_data("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/adni_tabular_v2.pkl")
# X = get_data("/Users/jieun/Dropbox/Desktop/UNC/Spring2021/COMP790/FinalProject/TCGA-GBM_cate2.pkl")


def impute_mean(X, mask):
    return fancyimpute_new(SimpleFill,{'fill_method':["mean"]}, X, mask)

def impute_knn(X, mask, hyperparams={'k':[2,4,6]}):
    return fancyimpute_new(KNN,hyperparams, X, mask)

def impute_mf(X, mask, hyperparams={'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}):
    return fancyimpute_new(MatrixFactorization, hyperparams, X, mask) 

def mse_acc(X_na, X_real, X_imputed, cat_index, cat=True):
    categorical_features = [str(i) for i in cat_index]
    numerical_features = set(X_na.columns) - set(categorical_features)
    num_miss = X_na[numerical_features]
    num_real = X_real[numerical_features]
    num_imp = X_imputed[numerical_features]
    
    # calculate MSE
    missing = pd.Series(num_miss.values.ravel('F'))
    real = pd.Series(num_real.values.ravel('F'))
    imputed = pd.Series(num_imp.values.ravel('F'))

    mse = evaluate_mse(real[missing.isnull()], imputed[missing.isnull()])
    # number of missing values
    n_nan = len(np.array(real[missing.isnull()]))
    # total mse = number of missing values * mse of view
    tmse = n_nan * mse
    ## assign 0 acc for numeric
    acc = 0

    if cat==True:
        ## categorical
        cat_miss = X_na[categorical_features]
        cat_real = X_real[categorical_features]
        cat_imp = X_imputed[categorical_features]
    
        # calculate AUC
        missing = pd.Series(cat_miss.values.ravel('F'))
        real = pd.Series(cat_real.values.ravel('F'))
        imputed = pd.Series(cat_imp.values.ravel('F'))

        y_true = np.array(real[missing.isnull()])
        y_imputed = np.array(imputed[missing.isnull()])
        acc = accuracy_score(y_true, y_imputed)

    return [tmse, n_nan, acc]

def generate_result(v, cat, view, categorical_index, numerical_index):
    view_na = view.copy()
    if cat==True:
        for subject in range(view.shape[0]):
                    view_na.loc[subject, view_mask[subject]] = 999
    else:
        for subject in range(view.shape[0]):
                    view_na.loc[subject, view_mask[subject]] = np.nan
    view_na_original = view_na.copy()

    if cat==True:
        ## Change numerical missing to np.nan
        for i_num_feat in numerical_index:
            na_ind = [x for x in list(range(view_na.shape[0])) if view_na.loc[x,[i_num_feat]].values in [999]] 
            view_na.loc[na_ind, [i_num_feat]] = np.nan
        ## Change categorical missing to np.nan
        list_of_binary_cat_feat = []
        for i_cat_feat in categorical_index:
            current_cat = view_na[i_cat_feat].values
            encoder_onehot = OneHotEncoder()
            encoder_label = LabelEncoder()
            encoded_labels = encoder_label.fit_transform(current_cat)[:, None]
            encoded_labels = encoder_onehot.fit_transform(encoded_labels).toarray()
            encoded_labels = pd.DataFrame(encoded_labels)
            if set(view_na_original[i_cat_feat].unique()).isdisjoint({999}):
                for i_col in range(encoded_labels.shape[1]):
                    view_na.insert(view_na.shape[1], str(i_cat_feat) + '_' + str(i_col), encoded_labels.loc[:, i_col])
                    list_of_binary_cat_feat.append(str(i_cat_feat) + '_' + str(i_col))
                print("Missing value changed from 999 to NaN")
            else:
                real_classes = encoded_labels.shape[1] - 1 
                for i in range(real_classes):
                    encoded_labels[i][encoded_labels[encoded_labels.shape[1]-1]==1] = np.nan
                encoded_labels.drop(encoded_labels.shape[1]-1, axis=1, inplace=True)
                encoded_labels = encoded_labels.to_numpy()
                for i_col in range(encoded_labels.shape[1]):
                    view_na.insert(view_na.shape[1], str(i_cat_feat) + '_' + str(i_col), encoded_labels[:, i_col])
                    list_of_binary_cat_feat.append(str(i_cat_feat) + '_' + str(i_col))
                print("Missing value changed from 999 to NaN")
            print("Categorical feature _", i_cat_feat, "_ completed")
        # delete cat original_features
        for i_cat_feat in categorical_index:
            view_na.drop(i_cat_feat, axis=1, inplace=True)
        print("NOTE: Categorical feature binarized")

    view_imputed = fancyimputer(**params).fit_transform(view_na)
    view_imputed = pd.DataFrame(view_imputed, columns=[str(col) for col in list(view_na.columns)])
    print("NOTE: Data was imputed")

    categorical_features = [str(i) for i in categorical_index]
    numerical_features = set(view_na_original.columns) - set(categorical_features)
    
    if cat==True:
        ## Accuracy Check
        for i_cat_feat in categorical_index:
            cat_i = [col for col in list(view_imputed.columns) if str(i_cat_feat) + '_' in col]
            final_cat = []
            for subject in range(view_imputed.shape[0]):
                val = view_imputed[cat_i].loc[subject,:]
                max_name = val.index.map(str)[val==max(val)][0]
                max_name = max_name.split("_",1)[1]
                final_cat.append(int(max_name))
            view_imputed.insert(view_imputed.shape[1]-1, str(i_cat_feat),  final_cat)
        # delete cat binary features
        for i_bin_feat in list_of_binary_cat_feat:
            view_imputed.drop(i_bin_feat, axis=1, inplace=True)
        
        view_na_original.columns = [str(col) for col in list(view_na_original.columns)]
        view.columns = [str(col) for col in list(view.columns)]
        # view_imputed = view_imputed[view_na_original.columns]
        view_na_original[view_na_original==999] = np.nan
        globals()["view" + str(v) + "_param" + str(pind) + "_imputed"] = view_imputed
#                 view_imputed.to_csv(get_name(fancyimputer) + "_param" + str(pind) + "_view" + str(v) + "_imputed_res"+ str(n_iter) +".csv")
        print("imputed data saved -- " + "view" + str(v) + "_param" + str(pind) + "_imputed")

        ## categorical
        cat_inc = view_na_original[categorical_features]
        cat_real = view[categorical_features]
        cat_imp = view_imputed[categorical_features]
        # calculate AUC
        a = pd.Series(cat_inc.values.ravel('F'))
        b = pd.Series(cat_real.values.ravel('F'))
        c = pd.Series(cat_imp.values.ravel('F'))
        y_true = np.array(b[a.isnull()])
        y_imputed = np.array(c[a.isnull()])
        acc = accuracy_score(y_true, y_imputed)
    else:
        acc = 0

    ## numeric
    num_inc = view_na_original[numerical_features]
    num_real = view[numerical_features]
    num_imp = view_imputed[numerical_features]
    # calculate MSE
    a = pd.Series(num_inc.values.ravel('F'))
    b = pd.Series(num_real.values.ravel('F'))
    c = pd.Series(num_imp.values.ravel('F'))
    mse = evaluate_mse(b[a.isnull()], c[a.isnull()])
    # number of missing values
    n_nan = len(np.array(b[a.isnull()]))
    # total mse = number of missing values * mse of view
    tmse = n_nan * mse

    return [v, tmse, n_nan, acc]


def fancyimpute_new(fancyimputer, param_candidates, X, n_iter, percent_missing=0.1):
    # first generate all parameter candidates for grid search
    all_param_candidates = dict_product(param_candidates)
    X_orig = X.copy()
    X_views = X[0]['X']
    X_catind = X[0]['CatogoricalIndicator']
    res_overall = pd.DataFrame(columns=['rmse','acc'])
    
    # Loop for each parameter candidates
    for params in all_param_candidates:
        print("Iteration start for: ", params)
        pind = [i for i in range(len(all_param_candidates)) if all_param_candidates[i]==params]
        res_view = pd.DataFrame(columns=['view_num', 'tmse', 'n_nan', 'acc'])
        
        ## Loop for each view
        for v in range(len(X_views)):
            print("View number :", v)
            view_cat = X_catind[v]
            view = X_views[v]
            view_mask = get_sn(view, percent_missing)
            ## scale all the numerical features to mean 0 sd 1
            
            # cols = set(np.array(range(len(view_cat)))[list(map(operator.not_,map(bool,view_cat)))])
            categorical_index = set(np.array(range(len(view_cat)))[list(map(bool,view_cat))])
            numerical_index = set(range(len(view_cat))) - categorical_index
            # convert view to dataframe
            view = view.tolist()
            view = pd.DataFrame(np.vstack(view))
            # view.columns = [str(i) for i   in view.columns]
            # scale
            for c in numerical_index:
                view[c]= pd.DataFrame(Normalize(pd.DataFrame(view[c])))
            
            ## Indicate if view has categorical feature(s)
            if sum(view_cat) !=0: cat=True
            else: cat=False
            print("Data type contains categorical?: ", cat)

            res_bridge = generate_results(v, cat, view, categorical_index, numerical_index)

            ## save results
            res = pd.DataFrame([res_bridge])
            res_view = res_view.append(res)
            print("Iteration end for: ", params)

        ## add mse for all      
        res_all = pd.DataFrame(res_view.sum(axis=0)).transpose()
        res_all['rmse'] = sqrt(res_all['tmse']/res_all['n_nan'])

    # best = np.array(res_all[['rmse']]).argmin()

    ## append all  MSEs and AUCs
    res_overall = res_overall.append(res_all[['rmse','acc']])
    res_overall.index = range(res_overall.shape[0])
    ## choose best parameter
    best = np.array(res_overall[['rmse']]).argmin()
    best_param = all_param_candidates[best]
        
    for v in range(len(X_views)):
        fin_impute = globals()["view" + str(v) + "_param" + str(pind) + "_imputed"]
        fin_impute.to_csv(get_name(fancyimputer) + "miss" + str(percent_missing) + "_param" + str(best) + "_view" + str(v) + "_imputed_res"+ str(n_iter) +".csv")

    final_mse = res_overall.loc[best,['rmse','acc']].values
    return res_overall


def results4method(method, params, data, iterations, missingness):
    for iter in range(iterations):
        iter_result = iter_result.append(fancyimpute_new(method, params, data, iter, missingness))
        print("iteration " + str(iter) + " done.")
    return iter_result

adni_mean10 = results4method(SimpleFill, {'fill_method':["mean"]}, X, 5, 0.1)
adni_mean20 = results4method(SimpleFill, {'fill_method':["mean"]}, X, 5, 0.2)
adni_mean30 = results4method(SimpleFill, {'fill_method':["mean"]}, X, 5, 0.3)
adni_mean40 = results4method(SimpleFill, {'fill_method':["mean"]}, X, 5, 0.4)
adni_mean50 = results4method(SimpleFill, {'fill_method':["mean"]}, X, 5, 0.5)

adni_knn10 = results4method(KNN,{'k':[2,4,6]}, X, 5, 0.1)
adni_knn20 = results4method(KNN,{'k':[2,4,6]}, X, 5, 0.2)
adni_knn30 = results4method(KNN,{'k':[2,4,6]}, X, 5, 0.3)
adni_knn40 = results4method(KNN,{'k':[2,4,6]}, X, 5, 0.4)
adni_knn50 = results4method(KNN,{'k':[2,4,6]}, X, 5, 0.5)

adni_mf10 = results4method(MatrixFactorization, {'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}, X, 5, 0.1)
adni_mf20 = results4method(MatrixFactorization, {'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}, X, 5, 0.2)
adni_mf30 = results4method(MatrixFactorization, {'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}, X, 5, 0.3)
adni_mf40 = results4method(MatrixFactorization, {'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}, X, 5, 0.4)
adni_mf50 = results4method(MatrixFactorization, {'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}, X, 5, 0.5)



















# 	
