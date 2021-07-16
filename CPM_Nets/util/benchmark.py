import numpy as np
import pandas as pd
from util import read_data
import itertools
from fancyimpute import (
    KNN,
    SimpleFill,
MatrixFactorization,
)
from get_sn import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()

def dict_product(hp_dict):
    '''
    Returns cartesian product of hyperparameters
    '''
    return [dict(zip(hp_dict.keys(),vals)) for vals in \
            itertools.product(*hp_dict.values())]

def fancyimpute_hpo(fancyimputer, param_candidates, X, mask, percent_validation=10):
    # first generate all parameter candidates for grid search
    all_param_candidates = dict_product(param_candidates)
    # get linear indices of all training data points
    train_idx = (mask.reshape(np.prod(X.shape)) == False).nonzero()[0]
    # get the validation mask
    n_validation = int(len(train_idx) * percent_validation/100)
    validation_idx = np.random.choice(train_idx,n_validation)
    validation_mask = np.zeros(np.prod(X.shape))
    validation_mask[validation_idx] = 1
    validation_mask = validation_mask.reshape(X.shape) > 0
    # save the original data
    X_incomplete = X.copy()
    # set validation and test data to nan
    X_incomplete[mask | validation_mask] = np.nan
    mse_hpo = []
    '''
    for params in all_param_candidates:
        if fancyimputer.__name__ != 'SimpleFill':
            params['verbose'] = False
        X_imputed = fancyimputer(**params).fit_transform(X_incomplete)
        mse = evaluate_mse(X_imputed, X, validation_mask)
        print(f"Trained {fancyimputer.__name__} with {params}, mse={mse}")
        mse_hpo.append(mse)
    '''
    best_params = all_param_candidates[0]
    # now retrain with best params on all training data
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    X_imputed = fancyimputer(**best_params).fit_transform(X_incomplete)
    mse_best = evaluate_mse(X_imputed, X, mask)
    print(f"HPO: {fancyimputer.__name__}, best {best_params}, mse={mse_best}")

    return X_imputed

def impute_mean(X, mask):
    return fancyimpute_hpo(SimpleFill,{'fill_method':["mean"]}, X, mask)

def impute_knn(X, mask, hyperparams={'k':[2,4,6]}):
    return fancyimpute_hpo(KNN, hyperparams, X, mask)
def impute_MF(X, mask, hyperparams={'rank':[10,],'l2_penalty':[1e-5], 'epochs': [100]}):
    return fancyimpute_hpo(MatrixFactorization, hyperparams, X, mask)


def SVM_eval(features, labels):
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





################### starts #############################



imputed_mean = {}
imputed_knn = {}
imputed_MF = {}
view_num = 1
acc_knn = {}
acc_mean = {}
missing_rate = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
for i in range(5):
    for i_missing in missing_rate:
        #Sn_all = get_sn(allData, view_num, allData.num_examples, i_missing)
        allData, trainData, testData, view_num = \
            read_data('/playpen-raid/data/oct_yining/multimp/data/adni_tabular_v2.pkl',
                      Normal=1,
                      multi_view=False, missing_rate=i_missing)
        Sn_all = allData.Sn_both
        for i in allData.data.keys():
            #Sn_all[i] = \
            #    np.logical_xor(np.logical_not(allData.Sn[i]),
            #                   np.logical_not(allData.MX[i])).astype('bool')[:, np.logical_not(allData.cat_indicator[i])]
            data = allData.data_both[i] #[:, np.logical_not(allData.cat_indicator[i])]
            imputed_knn[i] = impute_knn(data, np.logical_not(Sn_all[i]), hyperparams={'k': [2]})
            imputed_MF[i] = impute_MF(data, np.logical_not(Sn_all[i]))
            imputed_mean[i] = impute_mean(data, np.logical_not(Sn_all[i]))



        pca = PCA(n_components=128)
        X_pca_MF = pca.fit_transform(imputed_MF['0'])
        X_pca_mean = pca.fit_transform(imputed_mean['0'])
        X_pca_knn = pca.fit_transform(imputed_knn['0'])

        #acc_mean[i_missing] = SVM_eval(imputed_mean['0'], allData.labels)
        #acc_knn[i_missing] = SVM_eval(imputed_knn['0'], allData.labels)
        acc_mean = SVM_eval(X_pca_mean, allData.labels)
        acc_knn = SVM_eval(X_pca_knn, allData.labels)
        acc_MF = SVM_eval(X_pca_MF, allData.labels)
        #
        current_metrics_MF = dict()
        current_metrics_MF['data'] = ['ADNI']
        current_metrics_MF['missing'] = [i_missing*100]
        current_metrics_MF['accuracy'] = [acc_MF]
        current_metrics_MF['method'] = ['MatrixFactorization']
        #
        current_metrics_mean = dict()
        current_metrics_mean['data'] = ['ADNI']
        current_metrics_mean['missing'] = [i_missing*100]
        current_metrics_mean['accuracy'] = [acc_mean]
        current_metrics_mean['method'] = ['Mean']
        #current_metrics_mean['run_idx'] = [i_ri]
        current_metrics_knn = dict()
        current_metrics_knn['data'] = ['ADNI']
        current_metrics_knn['missing'] = [i_missing*100]
        current_metrics_knn['accuracy'] = [acc_knn]
        current_metrics_knn['method'] = ['KNN']
        print('acc for mean imputation: ' + str(acc_mean))
        print('acc for knn imputation: ' + str(acc_knn))
        print('acc for MF imputation: ' + str(acc_MF))
        ####
        metric_file = '/playpen-raid/data/oct_yining/multimp/results/metrics/adni_acc_comparisons_1.csv'
        ####################
        if os.path.exists(metric_file):
            metrics = pd.read_csv(metric_file, header=0)
            new_metrics = metrics.append(pd.DataFrame(current_metrics_mean,
                                                      columns=['data', 'missing', 'accuracy', 'method'],
                                                      index=[0]))
            new_metrics = new_metrics.append(pd.DataFrame(current_metrics_knn,
                                                      columns=['data', 'missing', 'accuracy', 'method'],
                                                      index=[0]))
            new_metrics = new_metrics.append(pd.DataFrame(current_metrics_MF,
                                                      columns=['data', 'missing', 'accuracy', 'method'],
                                                      index=[0]))
            new_metrics.to_csv(metric_file, index=None)
        else:
            pd.DataFrame.from_dict(current_metrics_mean).to_csv(metric_file, index=None)
            metrics = pd.read_csv(metric_file, header=0)
            new_metrics = metrics.append(pd.DataFrame(current_metrics_knn,
                                                      columns=['data', 'missing', 'accuracy', 'method'],
                                                      index=[0]))
            new_metrics = new_metrics.append(pd.DataFrame(current_metrics_MF,
                                                      columns=['data', 'missing', 'accuracy', 'method'],
                                                      index=[0]))
            new_metrics.to_csv(metric_file, index=None)
