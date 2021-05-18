import os
missing_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0]
for i in range(5):
    for im in missing_rate:
        command = 'python test_adni.py --model CPMNets_ori  --multi-view True --epochs-train 100 --missing-rate ' + str(im)
        os.system(command)