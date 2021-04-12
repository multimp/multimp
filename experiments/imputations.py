def impute_mean(X, mask):
    return fancyimpute_hpo(SimpleFill,{'fill_method':["mean"]}, X, mask)

def impute_knn(X, mask, hyperparams={'k':[2,4,6]}):
    return fancyimpute_hpo(KNN,hyperparams, X, mask)

def impute_mf(X, mask, hyperparams={'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}):
    return fancyimpute_hpo(MatrixFactorization, hyperparams, X, mask)


def impute_sklearn_rf(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    reg = RandomForestRegressor(random_state=0)
    parameters = {
        'n_estimators': [2, 10, 100],
        'max_features;': [int(np.sqrt(X.shape[-1])), X.shape[-1]]
                }
    clf = GridSearchCV(reg, parameters, cv=5)
    X_pred = IterativeImputer(random_state=0, predictor=reg).fit_transform(X_incomplete)
    mse = evaluate_mse(X_pred, X, mask)
    return mse

def impute_sklearn_linreg(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    reg = LinearRegression()
    X_pred = IterativeImputer(random_state=0, predictor=reg).fit_transform(X_incomplete)
    mse = evaluate_mse(X_pred, X, mask)
    return mse

def impute_datawig(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    df = pd.DataFrame(X_incomplete)
    df.columns = [str(c) for c in df.columns]
    dw_dir = os.path.join(DIR_PATH,'datawig_imputers')
    df = SimpleImputer.complete(df, output_path=dw_dir, hpo=True, verbose=0, iterations=1)
    for d in glob.glob(os.path.join(dw_dir,'*')):
        shutil.rmtree(d)
    mse = evaluate_mse(df.values, X, mask)
    return mse


def impute_datawig_iterative(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    df = pd.DataFrame(X_incomplete)
    df.columns = [str(c) for c in df.columns]
    df = SimpleImputer.complete(df, hpo=False, verbose=0, iterations=5)
    mse = evaluate_mse(df.values, X, mask)
    return mse