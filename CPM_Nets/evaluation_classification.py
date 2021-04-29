from util.evaluation import batch_evaluation
root = '/playpen-raid/data/oct_yining/multimp/results/imputed/'
result_path = '/playpen-raid/data/oct_yining/multimp/results/metrics/results.csv'
save_path = '/playpen-raid/data/oct_yining/multimp/results/metrics/results+acc.csv'
batch_evaluation(root, result_path, save_path)