import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
tf.compat.v1.disable_eager_execution()

class CPMNets_num_ori():
    """build model
    """
    def __init__(self, view_num, idx_record, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        # initialize parameter
        self.idx_record = idx_record
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        # initialize latent space data
        self.h_train, self.h_train_update = self.H_init('train')
        self.h = self.h_train
        self.h_index = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='h_index')
        self.h_temp = tf.gather_nd(self.h, self.h_index)
        # initialize the input data
        self.input = dict()
        self.sn = dict()
        self.output = dict()
        for v_num in range(self.view_num):
            self.input[str(v_num)] = tf.compat.v1.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                                    name='input' + str(v_num))
            self.sn[str(v_num)] = \
                tf.compat.v1.placeholder(tf.bool, shape=[None, self.layer_size[v_num][-1]], name='sn' + str(v_num))

            self.output[str(v_num)] = \
                tf.compat.v1.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                         name='output' + str(v_num))
        # ground truth
        self.gt = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='gt')
        # bulid the model
        self.train_op, self.loss = self.bulid_model([self.h_train_update, ], learning_rate)
        # open session
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def bulid_model(self, h_update, learning_rate):
        # initialize network
        net = dict()
        self.current_decoder = dict()
        for v_num in range(self.view_num):
            net[str(v_num)] = self.Encoding_net(self.h_temp, v_num)
            self.output[str(v_num)] = self.decoder_net(self.h_temp, v_num)
        # calculate reconstruction loss
        reco_regr_loss, reco_cls_loss = self.reconstruction_loss(net)
        # calculate classification loss
        all_loss = reco_regr_loss
        # train net operator
        # train the network to minimize reconstruction loss
        train_net_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
            .minimize(all_loss, var_list=tf.compat.v1.get_collection('weight'))
        # train the latent space data to minimize reconstruction loss and classification loss
        train_hn_op = tf.compat.v1.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=h_update[0])

        return [train_net_op, train_hn_op], [reco_regr_loss, all_loss]

    def H_init(self, a):
        with tf.compat.v1.variable_scope('H' + a):
            if a == 'train':
                h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim))
            elif a == 'test':
                h = tf.Variable(xavier_init(self.testLen, self.lsd_dim))
            h_update = tf.compat.v1.trainable_variables(scope='H' + a)
        return h, h_update

    def Encoding_net(self, h, v):
        weight = self.initialize_weight(self.layer_size[v])

        self.current_decoder[str(v)] = weight
        layer = tf.matmul(h, weight['w0']) + weight['b0']
        for num in range(1, len(self.layer_size[v])):
            layer = tf.nn.relu(layer)
            layer = tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)]
        layer = tf.math.tanh(layer)
        return layer

    def decoder_net(self, h, v):
        layer = tf.matmul(h, self.current_decoder[str(v)]['w0']) + self.current_decoder[str(v)]['b0']
        for num in range(1, len(self.layer_size[v])):
            layer = tf.nn.relu(layer)
            layer = tf.matmul(layer, self.current_decoder[str(v)]['w' + str(num)]) + self.current_decoder[str(v)]['b' + str(num)]
        layer = tf.math.tanh(layer)
        return layer

    def initialize_weight(self, dims_net):
        all_weight = dict()
        with tf.compat.v1.variable_scope('weight'):
            all_weight['w0'] = tf.Variable(xavier_init(self.lsd_dim, dims_net[0]))
            all_weight['b0'] = tf.Variable(tf.zeros([dims_net[0]]))
            tf.compat.v1.add_to_collection("weight", all_weight['w' + str(0)])
            tf.compat.v1.add_to_collection("weight", all_weight['b' + str(0)])
            for num in range(1, len(dims_net)):
                all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
                all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net[num]]))
                tf.compat.v1.add_to_collection("weight", all_weight['w' + str(num)])
                tf.compat.v1.add_to_collection("weight", all_weight['b' + str(num)])
        return all_weight

    def reconstruction_loss(self, net):
        loss_regr = 0
        loss_cls = 0
        for i_view in self.input.keys():
            # regression for numerical features
            loss_from_numeric_vs = tf.reduce_sum(
                tf.boolean_mask(tf.pow(tf.subtract(net[i_view], self.input[i_view]),
                                                    2.0), self.sn[i_view]))
            loss_regr += loss_from_numeric_vs
        return loss_regr, loss_cls

    def train(self, data, sn, gt, epoch, step=[5, 5]):
        global Reconstruction_LOSS

        for iter in range(epoch):
            index = np.array([x for x in range(self.trainLen)])
            shuffle(index)
            feed_dict = {self.input[str(v_num)]:
                             data[str(v_num)][index] + np.random.normal(size=data[str(v_num)][index].shape)*0.01
                         for v_num in range(self.view_num)}
            feed_dict.update({self.sn[str(i)]: sn[str(i)][index] for i in range(self.view_num)})
            feed_dict.update({self.gt: gt[index]})
            feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})

            # update the network
            for i in range(step[0]):
                _, Reconstruction_LOSS, Classification_LOSS = self.sess.run(
                    [self.train_op[0], self.loss[0], self.loss[1]], feed_dict=feed_dict)

            # update the h
            for i in range(step[1]):
                _, Reconstruction_LOSS, Classification_LOSS = self.sess.run(
                    [self.train_op[1], self.loss[0], self.loss[1]], feed_dict=feed_dict)

            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Classification Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, Classification_LOSS)
            print(output)

    def test(self, data, sn, gt, epoch):
        feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[str(i)] for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index:
                              np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})
        for iter in range(epoch):
            # update the network
            for i in range(5):
                _, Reconstruction_LOSS = self.sess.run(
                    [self.train_op[2], self.loss[0]], feed_dict=feed_dict)

            # update the h
            for i in range(5):
                _, Reconstruction_LOSS = self.sess.run(
                    [self.train_op[2], self.loss[0]], feed_dict=feed_dict)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" \
                .format((iter + 1), Reconstruction_LOSS)
            print(output)

    def recover(self, data, sn, gt):
        feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[str(i)] for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index:
                              np.array([x for x in range(self.trainLen)]).reshape(self.trainLen, 1)})
        imputed = {}
        for i in range(int(self.view_num)):
            imputed[str(i)] = self.sess.run(self.output[str(i)], feed_dict=feed_dict)
        return imputed

    def get_h_train(self):
        lsd = self.sess.run(self.h_train)
        return lsd[0:self.trainLen]

    def get_h_test(self):
        lsd = self.sess.run(self.h)
        return lsd[self.trainLen:]

    def get_h_all(self):
        lsd = self.sess.run(self.h_train)
        return lsd
