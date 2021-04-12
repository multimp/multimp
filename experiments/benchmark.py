'''
Benckmark

1. Zero Imputation
2. Mean Imputation
3. KNN
4. ...


'''

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

from sklearn.datasets import (
    make_low_rank_matrix,
    load_diabetes,
    load_wine,
    make_swiss_roll,
    load_breast_cancer,
    load_linnerud,
    load_boston
)

from fancyimpute import (
    MatrixFactorization,
    IterativeImputer,
    BiScaler,
    KNN,
    SimpleFill
)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
DIR_PATH = '.'

# this appears to be neccessary for not running into too many open files errors
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))


# 1. read data
# 2. data imputation
# 3. MSE
# 4. classification and evaluation

experiment()