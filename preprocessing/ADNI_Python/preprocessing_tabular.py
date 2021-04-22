'''
Arrange and save gene expression and clinical data as mat file according to views.

Yining Jiao
2021.4.15
'''


import pandas as pd
import numpy as np
import scipy.io as sio
import re
from sklearn.preprocessing import LabelEncoder
import pickle


# 1. read
adnimerge = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/tabular data/ADNIMERGE.csv'
gene_expression = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/ADNI_Gene_Expression_Profile/ADNI_Gene_Expression_Profile.csv'
df_tabular_data = pd.read_csv(adnimerge)
df_gene_expression = pd.read_csv(gene_expression, skiprows=[0, 1, 3, 4, 5, 6, 7, 8]).iloc[:, 3:-1].T

# 1. set names of each sample
names_of_samlpes = []
list_subjID = []
for i_sample in range(len(df_tabular_data)):
    current_name = str(df_tabular_data.RID[i_sample]) + '_' + str(df_tabular_data.VISCODE[i_sample])
    names_of_samlpes.append(current_name)
    list_subjID.append(df_tabular_data.PTID[i_sample])
df_tabular_data.insert(df_tabular_data.shape[1], 'SampleID', names_of_samlpes)

# 2. read gene expression
dict_person_genes = dict()
list_name_gene = []
for ith_patient in df_gene_expression.index:
    dict_person_genes[ith_patient] = df_gene_expression.loc[ith_patient].astype('float')
    list_name_gene.append(ith_patient)
df_gene = pd.DataFrame.from_dict(dict_person_genes, orient='index')
# 3. pair different views


dict_clinical_data = dict()
dict_person_DXbl = dict()
dict_person_DX = dict()
for i_subj_gene in list_name_gene:
    current_clinical_data = df_tabular_data[df_tabular_data.PTID==i_subj_gene]
    current_viscode = current_clinical_data.VISCODE
    whether_1st_exists = 0
    for i_vscode in current_viscode:
         #re.findall("\d+", current_viscode)[0]
        if i_vscode == 'bl':
            whether_1st_exists = 1
            dict_clinical_data[i_subj_gene] = \
                current_clinical_data[current_clinical_data.VISCODE==i_vscode].values.squeeze()
            dict_person_DXbl[i_subj_gene] = \
                current_clinical_data[current_clinical_data.VISCODE==i_vscode]['DX_bl'].values.squeeze()
            dict_person_DX[i_subj_gene] = \
                current_clinical_data[current_clinical_data.VISCODE==i_vscode]['DX'].values.squeeze()
    if whether_1st_exists == 0:
        print('There is no 1st scan for ' + str(i_subj_gene) + '.')

df_person_clinical = pd.DataFrame.from_dict(dict_clinical_data, orient='index', columns=df_tabular_data.keys())

# 4. delete features
delete_list = ['RID',
               'PTID',
               'VISCODE',
               'SITE',
               'COLPROT',
               'ORIGPROT',
               'EXAMDATE',
               'DX_bl',
               'DX',
               'update_stamp',
               'EXAMDATE_bl',
               'APOE4',]
for ith_feature in df_person_clinical.columns:
    if ith_feature + '_bl' in df_person_clinical.columns:
        delete_list.append(ith_feature + '_bl')
delete_list = np.unique(np.array(delete_list))

for i_del in delete_list:
    df_person_clinical.drop(i_del, axis=1, inplace=True)
for i_col in df_person_clinical.columns:
    NOT_ENOUGH_DATA = pd.isnull(df_person_clinical[i_col]).values.sum() > (len(df_person_clinical)//2)
    if df_person_clinical.dtypes[i_col] == 'object':
        try:
            df_person_clinical[i_col] = df_person_clinical[i_col].astype('float64')
        except:
            NOT_ENOUGH_DATA = True
            print('Data Type Error: ' + i_col)
    if NOT_ENOUGH_DATA:
        print('Delete columns: ' + str(i_col), ' lack ' + str(pd.isnull(df_person_clinical[i_col]).values.sum()) + ' values.')
        df_person_clinical.drop(i_col, axis=1, inplace=True)
# 5. save and concate
clinical_path = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/arranged/clinical.csv'
gene_path = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/arranged/gene.csv'
mat_path = 'E:/UNC-CS-Course/COMP 790-166/project/multimp/data/ADNI/arranged/adni_tabular.pkl'
df_person_clinical.to_csv(clinical_path)
df_gene.to_csv(gene_path)
views = dict()
views['X'] = dict()
views['X'] = np.array([tuple(np.array(df_person_clinical.values)), tuple(np.array(df_gene.values))], dtype="object")
labels = pd.DataFrame.from_dict(dict_person_DXbl, orient='index').values.squeeze().astype('str')
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
views['gt'] = encoded_labels[:, None]

#sio.savemat(mat_path, mdict={'X': (df_person_clinical.values, df_gene.values),
#                             'gt': encoded_labels[:, None]})
#np.save(mat_path, views)
with open(mat_path, 'wb') as handle:
    pickle.dump(views, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('processed')