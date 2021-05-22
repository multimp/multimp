from mvlearn.datasets import load_UCImultifeature
from mvlearn.cluster import MultiviewKMeans, MultiviewCoRegSpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as nmi_score, adjusted_rand_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
import pickle5 as pickle
# Load in UCI digits multiple feature dataset as an example
from util import read_data, DataSet
import numpy as np
from mvlearn.datasets import sample_joint_factor_model
from mvlearn.embed import CCA, MCCA, KMCCA, MVMDS
from mvlearn.plotting import crossviews_plot
from mvlearn.semi_supervised import CTClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import PCA
import pandas as pd


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

def transform_to_binaries(data, cat_indicator, labels):
    data_to_set = dict()
    for i_key in data.keys():
        data_to_set[int(i_key)] = data[i_key]
    view_number = len(data.keys())
    alldata = DataSet(data_to_set, view_number, np.array(labels), cat_indicator, multi_view=False,missing_rate=0)

    return alldata

RANDOM_SEED = 5

str_name = '/playpen-raid/data/oct_yining/multimp/data/TCGA-GBM_cate2.pkl'
str_name_adni = '/playpen-raid/data/oct_yining/multimp/data/adni_tabular_v2.pkl'

rootdir = '/playpen-raid/data/oct_yining/multimp/results/imputed/'
models = ['CPMNets_multiview_True',
          'CPMNets_multiview_False',
          'CPMNets_ori_multiview_True',
          'CPMNets_ori_multiview_False',
          'CPMNets_num_multiview_True',
          'CPMNets_num_multiview_False',
          'CPMNets_num_ori_multiview_True',
          'CPMNets_num_ori_multiview_False',]
missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
run_idx = [4] #[0, 1, 2, 3, 4, ]
metrics_file = '/playpen-raid/data/oct_yining/multimp/results/metrics/downstream_cls.csv'
for i_ri in run_idx:
    current_folder1 = os.path.join(rootdir, str(i_ri))
    for i_model in models:
        current_folder2 = os.path.join(current_folder1, i_model)
        for i_missing_rate in missing_rates:

            current_folder3 = os.path.join(current_folder2, str(i_missing_rate))
            filename = os.path.join(current_folder3, 'adni_missing_rate_' + str(i_missing_rate) + '.pkl')
            print(filename)
            multi_view = 'True' in i_model
            allData, trainData, testData, view_num = read_data('/playpen-raid/data/oct_yining/multimp/data/adni_tabular_v2.pkl',
                                                               Normal=1,
                                                               multi_view=multi_view,
                                                               missing_rate=0)
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            allData  = transform_to_binaries(data, allData.cat_indicator, allData.labels)


            pca = PCA(n_components=128)
            X_pca = pca.fit_transform(allData.data_both['0'])
            labels = allData.labels.squeeze()
            list_X_data = []
            for i_key in allData.data_both.keys():
                list_X_data.append(allData.data_both[i_key])
            acc = SVM_eval(X_pca, labels)
            current_metrics = dict()
            current_metrics['missing_rate'] = [i_missing_rate]
            current_metrics['acc'] = [acc]
            current_metrics['model'] = [i_model]
            current_metrics['run_idx'] = [i_ri]

            if os.path.exists(metrics_file):
                metrics = pd.read_csv(metrics_file, header=0)
                new_metrics = metrics.append(pd.DataFrame(current_metrics,
                                                          columns=['missing_rate', 'acc', 'model', 'run_idx'], index=[0]))
                new_metrics.to_csv(metrics_file, index=None)
            else:
                pd.DataFrame.from_dict(current_metrics).to_csv(metrics_file, index=None)


'''
pca = PCA(n_components=2)
pca.fit(X)
n_views = len(allData.data_both.keys())
n_samples = allData.data_both['0'].shape[0]
n_features = []
for i_key in allData.data_both.keys():
    n_features.append(allData.data_both[i_key].shape[1])
joint_rank = 3

cca = KMCCA(n_components=2, regs=0.01)
cca_scores = cca.fit_transform(list_X_data)

mvmds = MVMDS(n_components=100)
Xs_mvmds_reduced = mvmds.fit_transform(list_X_data)
acc = SVM_eval(Xs_mvmds_reduced, labels)
print('multiview acc Score: {0:.3f}\n'.format(acc))

ica = MultiviewICA(n_components=3, max_iter=10)
sources = ica.fit_transform(list_X_data)
acc = SVM_eval(sources, labels)
print('multiview acc Score: {0:.3f}\n'.format(acc))

crossviews_plot(cca_scores,
                title='CCA scores (first two views fitted)',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})


#######################################
m_kmeans = MultiviewKMeans(
    n_clusters=len(np.unique(labels)), random_state=RANDOM_SEED)

m_clusters = m_kmeans.fit_predict(list_X_data)

# Compute nmi between true class labels and multiview cluster labels
m_nmi = nmi_score(labels.squeeze(), m_clusters)
m_ari = adjusted_rand_score(labels.squeeze(), m_clusters)
print('multiview NMI Score: {0:.3f}\n'.format(m_nmi))
print('multiview ARI Score: {0:.3f}\n'.format(m_ari))


# Use the MultiviewSphericalKMeans instance to cluster the data
m_kmeans = MultiviewCoRegSpectralClustering(
    n_clusters=len(np.unique(labels)), n_init=10, max_iter=6,
    random_state=RANDOM_SEED)
m_clusters = m_kmeans.fit_predict(list_X_data)

# Compute nmi between true class labels and multiview cluster labels
m_nmi = nmi_score(labels.squeeze(), m_clusters)
m_ari = adjusted_rand_score(labels.squeeze(), m_clusters)
print('multiview NMI Score: {0:.3f}\n'.format(m_nmi))
print('multiview ARI Score: {0:.3f}\n'.format(m_ari))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import mvlearn as mv
from mvlearn.cluster import MultiviewSphericalKMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score



# 1. read data
rootdir = '/playpen-raid/data/oct_yining/multimp/data/results/imputed/'
dict_results = dict()
run_idx = [0, 1, 2, 3, 4]
#missing_rates = ['0', '0.1','0.2', '0.3', '0.4', '0.5']
missing_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
multi_view = [True, False]
models = ['CPMNets_num', 'CPMNets_num_ori', 'CPMNets', 'CPMNets_ori',]# ['CPMNets_num', 'CPMNets_num_ori']
metrics = ['mse', 'acc']

def get_name_of_imputed(rootdir, model, missingrate, multiview, run_idx):
    return name_of_imputed


dict_downstream_cls = dict()
for i_model in models:
    dict_downstream_cls[i_model] = dict()
    for i_mv in multi_view:
        dict_downstream_cls[i_model][i_mv] = dict()
        for i_msr in missing_rates:
            dict_downstream_cls[i_model][i_mv][i_msr] = dict()
            name_of_imputed = get_name_of_imputed(rootdir, i_model, i_msr, i_mv, 0)
            with open(name_of_imputed, 'rb') as handle:
                data = pickle.load(handle)
            dict_downstream_cls
            
'''