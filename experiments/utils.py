
def plot_results(results):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(open(os.path.join(DIR_PATH, 'benchmark_results.csv')))
    df['mse_percent'] = df.mse / df.groupby(['data','missingness','percent_missing'])['mse'].transform(max)
    df.groupby(['missingness','percent_missing','imputer']).agg({'mse_percent':'median'})

    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("RdBu_r", 7))
    sns.set_context("notebook",
                    font_scale=1.3,
                    rc={"lines.linewidth": 1.5})
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    sns.boxplot(hue='imputer',
                y='mse_percent',
                x='percent_missing', data=df[df['missingness']=='MCAR'])
    plt.title("Missing completely at random")
    plt.xlabel('Percent Missing')
    plt.ylabel("Relative MSE")
    plt.gca().get_legend().remove()


    plt.subplot(1,3,2)
    sns.boxplot(hue='imputer',
                y='mse_percent',
                x='percent_missing',
                data=df[df['missingness']=='MAR'])
    plt.title("Missing at random")
    plt.ylabel('')
    plt.xlabel('Percent Missing')
    plt.gca().get_legend().remove()

    plt.subplot(1,3,3)
    sns.boxplot(hue='imputer',
                y='mse_percent',
                x='percent_missing',
                data=df[df['missingness']=='MNAR'])
    plt.title("Missing not at random")
    plt.ylabel("")
    plt.xlabel('Percent Missing')

    handles, labels = plt.gca().get_legend_handles_labels()

    l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig('benchmarks_datawig.pdf')


def dict_product(hp_dict):
    '''
    Returns cartesian product of hyperparameters
    '''
    return [dict(zip(hp_dict.keys(),vals)) for vals in \
            itertools.product(*hp_dict.values())]

def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()

def get_data(data_fn):
    if data_fn.__name__ is 'make_low_rank_matrix':
        X = data_fn(n_samples=1000, n_features=10, effective_rank = 5, random_state=0)
    elif data_fn.__name__ is 'make_swiss_roll':
        X, t = data_fn(n_samples=1000, random_state=0)
        X = np.vstack([X.T, t]).T
    else:
        X, _ = data_fn(return_X_y=True)
    return X