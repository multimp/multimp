
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_result_ADNI_ours = 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/adni_downstream_cls_latent.csv'

# 1. read pickle
df_results = pd.read_csv(path_result_ADNI_ours)
dict_results = dict()
run_idx = [0, 1, 2, 3, 4]
#missing_rates = ['0', '0.1','0.2', '0.3', '0.4', '0.5']
missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
multi_view = [True, False]

models = [
    'CPMNets_num_multiview_True',
    'CPMNets_num_multiview_False',
    'CPMNets_num_ori_multiview_True',
    'CPMNets_num_ori_multiview_False',
    'CPMNets_multiview_True',
    'CPMNets_multiview_False',
    'CPMNets_ori_multiview_True',
    'CPMNets_ori_multiview_False',


]


name_transformer = {'CPMNets_multiview_True':'MultImp/CELoss/GAN/multi_view',
          'CPMNets_multiview_False':'MultImp/CELoss/GAN/single_view',
          'CPMNets_ori_multiview_True':'MultImp/CELoss/no GAN/multi_view',
          'CPMNets_ori_multiview_False':'MultImp/CELoss/no GAN/single_view',
          'CPMNets_num_multiview_True':'MultImp/no CELoss/GAN/multi_view',
          'CPMNets_num_multiview_False':'MultImp/no CELoss/GAN/single_view',
          'CPMNets_num_ori_multiview_True':'MultImp/no CELoss/no GAN/multi_view',
          'CPMNets_num_ori_multiview_False':'MultImp/no CELoss/no GAN/single_view',}

models_with_mv = []
for i_model_with_mv in models:
    current_results_0 = df_results[df_results['model']==i_model_with_mv]
    dict_results[i_model_with_mv] = dict()
    models_with_mv.append(i_model_with_mv)
    for i_missing_rate in missing_rates:

        dict_results[i_model_with_mv][i_missing_rate] = dict()
        current_accs = []
        current_results_1 = current_results_0[current_results_0['missing_rate']==i_missing_rate]
        for i_rd in run_idx:
            current_results_2 = current_results_1[current_results_1['run_idx']==i_rd]
            if len(current_results_2) >1:
                print('More than 1 result for the same setting')
            current_accs.append(current_results_2['acc'].values[-1])
        dict_results[i_model_with_mv][i_missing_rate]['acc_mean'] = np.array(current_accs).mean()
        dict_results[i_model_with_mv][i_missing_rate]['acc_std'] = np.array(current_accs).std()

'''
# 2. read comparisons
path_result_ADNI_ours = 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/rmse_acc_all.csv'
df_results_comparisons = pd.read_csv(path_result_ADNI_ours)
df_results_comparisons = df_results_comparisons[df_results_comparisons['data']=='ADNI']

models_com = ['Mean', 'KNN', 'MatrixFactorization']
for i_model in models_com:
    models_with_mv.append(i_model)
    current_results_0 = df_results_comparisons[df_results_comparisons['method']==i_model]
    dict_results[i_model] = dict()
    for i_missing_rate in missing_rates:
        dict_results[i_model][i_missing_rate] = dict()
        current_rmses = []
        current_accs = []
        current_results_1 = current_results_0[current_results_0['missingrate']==i_missing_rate]
        if len(current_results_1) >5:
            print('More than 5 result for the same setting in comparisons')
        for i_rd in range(len(current_results_1)):
            current_rmses.append(current_results_1['RMSE'].values.squeeze()[i_rd])
            current_accs.append(current_results_1['ACC'].values.squeeze()[i_rd])
        dict_results[i_model][i_missing_rate]['rmse_mean'] = np.array(current_rmses).mean()
        dict_results[i_model][i_missing_rate]['acc_mean'] = np.array(current_accs).mean()
        dict_results[i_model][i_missing_rate]['rmse_std'] = np.array(current_rmses).std()
        dict_results[i_model][i_missing_rate]['acc_std'] = np.array(current_accs).std()
'''


# 3. plot
colors = ['tomato', 'gold', 'cornflowerblue', 'springgreen', 'maroon', 'orange', 'navy', 'darkgreen', 'black', 'blueviolet', 'magenta']
dict_color_models_with_mv = dict()
for i in range(len(models_with_mv)):
    dict_color_models_with_mv[models_with_mv[i]] = colors[i]



def ACC_plot(which, savename_acc):
    ############### ACC ###############
    color_list = []
    for i in which:
        color_list.append(dict_color_models_with_mv[models_with_mv[i]])
    model_list = []
    for i in which:
        model_list.append(models_with_mv[i])

    plt.figure(dpi=500)

    ax = plt.gca()
    #ax.set_xlim(0, 402)
    #ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis =np.array(missing_rates)

    for i_model_with_mv, color in zip(model_list, color_list):
                                                               #'pink', 'yellow', 'cyan', 'darkgreen']):
        mean_list = []
        std_list = []
        for i_missing_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            mean_list.append(dict_results[i_model_with_mv][i_missing_rate]['acc_mean'])
            std_list.append(dict_results[i_model_with_mv][i_missing_rate]['acc_std'])

        ax.fill_between(X_axis, np.array(mean_list) - np.array(std_list),
                        np.array(mean_list) + np.array(std_list),
                        alpha=0.1, color=color)
        ax.plot(X_axis, np.array(mean_list), color=color,
                alpha=1, label=name_transformer[i_model_with_mv])

    plt.title("Accracy of Downtream Classification Task")
    plt.xlabel("Missing Rate")
    plt.ylabel("Accracy")
    plt.legend()
    plt.grid(False)
    plt.xticks(X_axis)
    plt.savefig(savename_acc)
    plt.show()

# overall
#ACC_plot([0, 1, 2, 3, 8, 9, 10], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/adni_acc_no_CELoss.png')
#ACC_plot([4, 5, 6, 7, 8, 9, 10], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/adni_acc_CELoss.png')

ACC_plot([0, 1, 2, 3], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_no_CELoss_1234.png')
ACC_plot([4, 5, 6, 7], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_CELoss_1234.png')

ACC_plot([0, 4],  'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_compare_1.png')
ACC_plot([1, 5], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_compare_2.png')
ACC_plot([2, 6],  'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_compare_3.png')
ACC_plot([3, 7], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_compare_4.png')

ACC_plot([0, 2, 4, 6],  'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_compare_merge_1.png')
ACC_plot([1, 3, 5, 7], 'E:/UNC-CS-Course/COMP 790-166/project/results/metrics/plots/adni/latent_adni_downstream_compare_merge_2.png')