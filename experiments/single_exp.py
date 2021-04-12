import os
import glob
import sys
import shutil
import json
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# sys.path.insert(0,'')
from datawig import SimpleImputer



np.random.seed(0)
DIR_PATH = '.'

def experiment(DATALOADERS, imputers, percent_missing_list=[10, 30], nreps = 3):

    '''
    DATA_LOADERS = [
        make_low_rank_matrix,
        load_diabetes,
        load_wine,
        make_swiss_roll,
        load_breast_cancer,
        load_linnerud,
        load_boston
    ]

    imputers = [
        impute_mean,
        impute_knn,
        impute_mf,
        impute_sklearn_rf,
        impute_sklearn_linreg,
        impute_datawig
    ]
    '''
    results = []
    with open(os.path.join(DIR_PATH, 'benchmark_results.json'), 'w') as fh:
        for percent_missing in tqdm(percent_missing_list):
            for data_fn in DATA_LOADERS:
                X = get_data(data_fn)
                for missingness in ['MCAR', 'MAR', 'MNAR']:
                    for _ in range(nreps):
                        missing_mask = generate_missing_mask(X, percent_missing, missingness)
                        for imputer_fn in imputers:
                            mse = imputer_fn(X, missing_mask)
                            result = {
                                'data': data_fn.__name__,
                                'imputer': imputer_fn.__name__,
                                'percent_missing': percent_missing,
                                'missingness': missingness,
                                'mse': mse
                            }
                            fh.write(json.dumps(result) + "\n")
                            print(result)

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
    for params in all_param_candidates:
        if fancyimputer.__name__ != 'SimpleFill':
            params['verbose'] = False
        X_imputed = fancyimputer(**params).fit_transform(X_incomplete)
        mse = evaluate_mse(X_imputed, X, validation_mask)
        print(f"Trained {fancyimputer.__name__} with {params}, mse={mse}")
        mse_hpo.append(mse)

    best_params = all_param_candidates[np.array(mse_hpo).argmin()]
    # now retrain with best params on all training data
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    X_imputed = fancyimputer(**best_params).fit_transform(X_incomplete)
    mse_best = evaluate_mse(X_imputed, X, mask)
    print(f"HPO: {fancyimputer.__name__}, best {best_params}, mse={mse_best}")
    return mse_best
