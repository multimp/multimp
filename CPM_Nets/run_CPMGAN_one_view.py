import os
missing_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0]
for i in range(1):
    for im in missing_rate:
        command = 'python test_adni.py ' \
                  '--model CPMNets ' \
                  '--multi-view 0 ' \
                  '--epochs-train 30 ' \
                  '--missing-rate ' + str(im)+  ' --run-idx ' + str(i)
        os.system(command)