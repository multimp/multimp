# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:34:18 2021

@author: wancenmu
"""

import pyreadr
import numpy as np
import pickle

result = pyreadr.read_r("C:/Users/wancenmu/Downloads/GBMLGG/TCGA-GBM_cate.rdata")

clinical = result['clinical_GBM3']
mutation = result['mutation_GBM2']
rna = result['rna_GBM2']
CNV = result['CNV_GBM2']
CNV_focal = result['CNV_focal_GBM2']
methylation = result['methylation_GBM2']
miRNA = result['miRNA_GBM2']
# read in category 
cate_clinical = result['cate_clinical']
cate_mutation = result['cate_mutation']
cate_rna = result['cate_rna']
cate_CNV = result['cate_CNV']
cate_CNV_focal = result['cate_CNV_focal']
cate_methylation = result['cate_methylation']
cate_miRNA = result['cate_miRNA']

# label
label = result['label']
status = result['status']
time = result['time']

views = dict()
views['X'] = dict()
views['X'] = \
    np.array([tuple(np.array(clinical)),
              tuple(np.array(mutation)),
              tuple(np.array(rna)),
              tuple(np.array(CNV)),
              tuple(np.array(CNV_focal)),
              tuple(np.array(methylation)),
              tuple(np.array(miRNA))], dtype="object")

views['CatogoricalIndicator'] = \
    np.array([tuple(np.array(cate_clinical).reshape(-1)),
              tuple(np.array(cate_mutation).reshape(-1)),
              tuple(np.array(cate_rna).reshape(-1)),
              tuple(np.array(cate_CNV).reshape(-1)),
              tuple(np.array(cate_CNV_focal).reshape(-1)),
              tuple(np.array(cate_methylation).reshape(-1)),
              tuple(np.array(cate_miRNA).reshape(-1))], dtype="object")

views['gt'] = np.array(label).astype(int)

mat_path = "C:/Users/wancenmu/Downloads/GBMLGG/TCGA-GBM_cate.pkl"

with open(mat_path, 'wb') as handle:
    pickle.dump(views, handle, protocol=pickle.HIGHEST_PROTOCOL)

