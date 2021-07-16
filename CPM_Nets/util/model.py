
import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()
from sklearn.metrics import roc_auc_score
class CPMNets():
    """build model
    """
    def __init__(self, view_num, idx_record, trainLen, testLen, layer_size, layer_size_d, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
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
        self.layer_size_d = layer_size_d
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        # initialize latent space data
        self.h_train, self.h_train_update = self.H_init('train')
        #self.h_test, self.h_test_update = self.H_init('test')
        self.h = self.h_train #tf.concat([self.h_train, self.h_test], axis=0)
        self.h_index = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='h_index')
        self.h_temp = tf.gather_nd(self.h, self.h_index)
        # initialize the input data
        self.input = dict()
        self.output = dict()
        self.sn = dict()
        for v_num in range(self.view_num):
            self.input[str(v_num)] = \
                tf.compat.v1.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]], name='input' + str(v_num))

            self.output[str(v_num)] = \
                tf.compat.v1.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                         name='output' + str(v_num))
            self.sn[str(v_num)] = \
                tf.compat.v1.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]], name='sn' + str(v_num))

        # ground truthlo
        self.gt = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='gt')

        # bulid the model
        #self.train_op, self.loss = self.bulid_model([self.h_train_update, self.h_test_update], learning_rate)
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

        # discriminator
        discriminator_net = dict()
        discriminator_labels = dict()

        for v_num in range(self.view_num):
            self.weight_d = self.initialize_weight_for_discr(self.layer_size_d[v_num])
            discriminator_net[str(v_num)], discriminator_labels[str(v_num)] = \
                self.Discriminator_net(self.input[str(v_num)], net[str(v_num)], v_num)

        # calculate reconstruction loss
        reco_loss_regre, reco_loss_cls = self.reconstruction_loss(net)

        # gan discriminator loss
        discriminator_loss = self.discriminator_loss(discriminator_net, discriminator_labels, label_smoothing=0)
        # gan generator loss
        generator_loss = self.generator_loss(discriminator_net, discriminator_labels, label_smoothing=0)
        # recons loss
        recons_loss = tf.add(reco_loss_regre, reco_loss_cls)
        # gan loss
        gan_loss = tf.add(discriminator_loss, generator_loss)
        # all loss
        all_loss = tf.add(recons_loss, gan_loss)

        # train net operator
        # train the network to minimize reconstruction loss and generator loss
        train_net_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
            .minimize(recons_loss, var_list=tf.compat.v1.get_collection('weight'))

        train_gen_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
            .minimize(generator_loss, var_list=tf.compat.v1.get_collection('weight'))

        # train the latent space data to minimize reconstruction loss and classification loss

        train_hn_op = tf.compat.v1.train.AdamOptimizer(learning_rate[1]) \
            .minimize(recons_loss, var_list=h_update[0])
        train_hn_gen_op = tf.compat.v1.train.AdamOptimizer(learning_rate[1]) \
            .minimize(generator_loss, var_list=h_update[0])

        # discriminator
        train_discriminator_op = tf.compat.v1.train.AdamOptimizer(learning_rate[1]) \
            .minimize(discriminator_loss, var_list=tf.compat.v1.get_collection('weight_discri'))


        return [train_net_op, train_gen_op,
                train_hn_op, train_hn_gen_op, #adj_hn_op,
                train_discriminator_op], \
               [all_loss, recons_loss, gan_loss,
                generator_loss, discriminator_loss,
                reco_loss_regre, reco_loss_cls]


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

    def Discriminator_net(self, x_real, x_generated, v):
        # concate and suffle data
        x_real = tf.gather(x_real * tf.cast(self.sn[str(v)], tf.float32),
                           indices=self.idx_record[str(v)]['value'],
                           axis=1) #+ x_generated * (1 -  tf.cast(self.sn[str(v)], tf.float32))
        x_generated = tf.gather(x_generated * tf.cast(self.sn[str(v)], tf.float32),
                                indices=self.idx_record[str(v)]['value'],
                                axis=1)

        x_feat = tf.concat((x_real, x_generated), axis=0)

        y = tf.cast(tf.concat((tf.ones_like(self.gt), tf.zeros_like(self.gt)), axis=0), tf.float32)

        layer_d = tf.matmul(x_feat, self.weight_d['w0']) + self.weight_d['b0']
        if len(self.layer_size_d[v]) >2:
            for num in range(1, len(self.layer_size_d[v])-1):
                layer_d = tf.nn.relu(layer_d)
                layer_d = tf.matmul(layer_d, self.weight_d['w' + str(num)]) + self.weight_d['b' + str(num)]
        return layer_d, y

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

    def initialize_weight_for_discr(self, dims_net_discr):
        all_weight = dict()
        with tf.compat.v1.variable_scope('weight_discri'):
            if len(dims_net_discr) > 1:
                for num in range(0, len(dims_net_discr)-1):
                    all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net_discr[num], dims_net_discr[num+1]))
                    all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net_discr[num+1]]))
                    tf.compat.v1.add_to_collection("weight_discri", all_weight['w' + str(num)])
                    tf.compat.v1.add_to_collection("weight_discri", all_weight['b' + str(num)])
        return all_weight


    def reconstruction_loss(self, net):
        loss_regr = 0
        loss_cls = 0
        for i_view in self.input.keys():
            # regression for numerical features
            reconst_val_i = tf.gather(net[i_view], indices=self.idx_record[i_view]['value'], axis=1)
            input_val_i = tf.gather(self.input[i_view], indices=self.idx_record[i_view]['value'], axis=1)
            sn_val_i = tf.gather(self.sn[i_view], indices=self.idx_record[i_view]['value'], axis=1)
            loss_from_numeric_vs = tf.reduce_sum(
            tf.boolean_mask(tf.pow(tf.subtract(reconst_val_i, input_val_i),
                                               2.0), sn_val_i))

            loss_regr += loss_from_numeric_vs

            # cls for categorical features
            if len(self.idx_record[i_view]['cat']) > 0:
                loss_from_cat_vs = 0.0
                for ith_cat in self.idx_record[i_view]['cat'].keys():

                    reconst_cat_i = tf.gather(net[i_view], indices=self.idx_record[i_view]['cat'][ith_cat], axis=1)
                    input_cat_i = tf.cast(tf.gather(self.input[i_view], indices=self.idx_record[i_view]['cat'][ith_cat], axis=1), tf.float32)
                    sn_cat_i = tf.gather(self.sn[i_view], indices=self.idx_record[i_view]['cat'][ith_cat], axis=1)
                    probs = tf.boolean_mask(tf.compat.v1.math.softmax(reconst_cat_i, axis=1), sn_cat_i)
                    cross_entropy = \
                        tf.compat.v1.math.log(probs + 1e-3) * tf.boolean_mask(input_cat_i, sn_cat_i)
                    loss_from_cat_vs -= tf.reduce_sum(cross_entropy)
                loss_cls += loss_from_cat_vs

        return loss_regr, loss_cls

    def discriminator_loss(self,
            discriminator_outputs,
            discriminator_labels,
            label_smoothing=0,
            weights=1.0,
            scope=None,
            loss_collection=tf.compat.v1.GraphKeys.LOSSES,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
            add_summaries=False):
        loss = 0

        with tf.compat.v1.name_scope(scope, 'discriminator_loss', ) as scope:
            for ith_view in range(int(self.view_num)):
                probs = tf.compat.v1.math.sigmoid(discriminator_outputs[str(ith_view)])
                cross_entropy = \
                    tf.compat.v1.math.log(probs+1e-3) * discriminator_labels[str(ith_view)]+ \
                    tf.compat.v1.math.log(1-probs+1e-3) * tf.subtract(1.0, discriminator_labels[str(ith_view)])
                loss -= tf.reduce_sum(cross_entropy)
            tf.compat.v1.losses.add_loss(loss, loss_collection)
            if add_summaries:
                tf.compat.v1.summary.scalar('discriminator_loss', loss)

        return loss

    def generator_loss(self,
            discriminator_outputs,
            discriminator_labels,
            label_smoothing=0.0,
            weights=1.0,
            scope=None,
            loss_collection=tf.compat.v1.GraphKeys.LOSSES,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
            add_summaries=False):
        loss = 0
        with tf.compat.v1.name_scope(scope, 'generator_loss') as scope:
            loss -= \
                self.discriminator_loss(
                discriminator_outputs,
                discriminator_labels,
                weights=weights,  scope=scope, label_smoothing=label_smoothing,
                loss_collection=loss_collection)
                #reduction=reduction)

        if add_summaries:
            tf.compat.v1.summary.scalar('generator_loss', loss)

        return loss



    def train(self, data, sn, gt, epoch, step=[5, 5, 5, 5, 5, 5]):
        global ReconstructionLoss, ClsLoss, GeneratorLoss, DiscriminatorLoss, AllLoss

        for iter in range(epoch):
            index = np.array([x for x in range(self.trainLen)])
            shuffle(index)
            feed_dict = {self.input[str(v_num)]:
                             data[str(v_num)][index] + np.random.normal(size=data[str(v_num)][index].shape)*0.01
                         for v_num in range(self.view_num)}
            feed_dict.update({self.sn[str(i)]: sn[str(i)][index] for i in range(self.view_num)})
            feed_dict.update({self.gt: gt[index]})
            feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})

            # updata the discriminator
            for i in range(step[4]):
                _, DiscriminatorLoss = self.sess.run(
                    [self.train_op[4], self.loss[4]], feed_dict=feed_dict)

            # update gan
            for i in range(step[1]):
                _, ReconstructionLoss, ClsLoss, GeneratorLoss = self.sess.run(
                    [self.train_op[1], self.loss[5], self.loss[6], self.loss[3]], feed_dict=feed_dict)
            # update gan
            for i in range(step[3]):
                _, AllLoss = self.sess.run(
                    [self.train_op[3], self.loss[0]], feed_dict=feed_dict)

            # update the network
            for i in range(step[0]):
                _, ReconstructionLoss, ClsLoss, GeneratorLoss = self.sess.run(
                    [self.train_op[0], self.loss[5], self.loss[6], self.loss[3]], feed_dict=feed_dict)

            # update the h
            for i in range(step[2]):
                _, AllLoss = self.sess.run(
                    [self.train_op[2], self.loss[0]], feed_dict=feed_dict)


            output = "Epoch : {:.0f}  ===> " \
                     "All Loss = {:.4f}, " \
                     "Reconstruction Loss = {:.4f}, " \
                     "Cls Loss = {:.4f}, " \
                     "Generator Loss = {:.4f}, " \
                     "Discriminator Loss = {:.4f}" \
                .format((iter + 1),
                        AllLoss,
                        ReconstructionLoss,
                        ClsLoss,
                        GeneratorLoss,
                        DiscriminatorLoss)
            print(output)


    def test(self, data, sn, gt, epoch):
        feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[str(i)] for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index:
                              np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1)})
        for iter in range(epoch):
            # update the h
            for i in range(5):
                _, Reconstruction_LOSS = self.sess.run(
                    [self.train_op[2], self.loss[1]], feed_dict=feed_dict)
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

