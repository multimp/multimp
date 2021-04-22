import numpy as np
from util.util import read_data
from util.get_sn import get_sn
from util.model import CPMNets
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=512,
                        help='dimensionality of the latent space data [default: 512]')
    parser.add_argument('--epochs-train', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 20]')
    parser.add_argument('--epochs-test', type=int, default=100, metavar='N',
                        help='number of epochs to test [default: 50]')
    parser.add_argument('--lamb', type=float, default=10.,
                        help='trade off parameter [default: 10]')
    parser.add_argument('--missing-rate', type=float, default=0.1,
                        help='view missing rate [default: 0]')
    args = parser.parse_args()

    # read data
    trainData, testData, view_num = read_data('/playpen-raid/data/oct_yining/multimp/data/adni_tabular.pkl', 0.8, 1)
    outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]
    # set layer size
    layer_size = [[outdim_size[i]] for i in range(view_num)]
    # set parameter
    epoch = [args.epochs_train, args.epochs_test]
    learning_rate = [0.001, 0.01]
    # Randomly generated missing matrix
    Sn_train = get_sn(trainData, view_num, trainData.num_examples, args.missing_rate)
    Sn_test = get_sn(testData, view_num, testData.num_examples, args.missing_rate)
    # Model building
    model = CPMNets(view_num, trainData.num_examples, testData.num_examples, layer_size, args.lsd_dim, learning_rate,
                    args.lamb)
    # train
    model.train(trainData.data, Sn_train, trainData.labels.reshape(trainData.num_examples), epoch[0])
    H_train = model.get_h_train()
    # test
    model.test(testData.data, Sn_test, testData.labels.reshape(testData.num_examples), epoch[1])
    H_test = model.get_h_test()
    label_pre = classfiy.ave(H_train, H_test, trainData.labels)
    print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre)))