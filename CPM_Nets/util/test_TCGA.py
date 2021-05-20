import pickle5 as pickle

path = '/playpen-raid/data/oct_yining/multimp/data/TCGA-GBM_cate2.pkl'
with open(path, 'rb') as handle:
    data = pickle.load(handle)
print('1')