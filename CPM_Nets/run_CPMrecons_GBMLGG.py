import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
missing_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
for i in range(5):
    for im in missing_rate:
        command = 'python test_GBMLGG2.py ' \
                  '--model CPMNets_ori  ' \
                  '--multi-view 1 ' \
                  '--epochs-train 100 ' \
                  '--missing-rate ' + str(im) + ' --run-idx ' + str(i)
        os.system(command)
