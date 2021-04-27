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
    def __init__(self, view_num, cat_indicator, trainLen, testLen, layer_size, layer_size_d, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        # initialize parameter
        self.cat_indicator = cat_indicator
        self.view_num = view_num
        self.layer_size = layer_size
        self.layer_size_d = layer_size_d
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        # initialize latent space data
        self.h_train, self.h_train_update = self.H_init('train')
        self.h_test, self.h_test_update = self.H_init('test')
        self.h = tf.concat([self.h_train, self.h_test], axis=0)
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
        self.train_op, self.loss = self.bulid_model([self.h_train_update, self.h_test_update], learning_rate)
        # open session
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def bulid_model(self, h_update, learning_rate):
        # initialize network
        net = dict()
        for v_num in range(self.view_num):
            net[str(v_num)] = self.Encoding_net(self.h_temp, v_num)
            self.output[str(v_num)] = net[str(v_num)]

        # discriminator
        discriminator_net = dict()
        discriminator_labels = dict()
        for v_num in range(self.view_num):
            discriminator_net[str(v_num)], current_labels = \
                self.Discriminator_net(self.input[str(v_num)], net[str(v_num)], v_num)
            discriminator_labels[str(v_num)] = current_labels


        # calculate reconstruction loss
        reco_loss_regre, reco_loss_cls = self.reconstruction_loss(net)

        # gan discriminator loss
        discriminator_loss = self.discriminator_loss(discriminator_net, discriminator_labels)
        # gan generator loss
        generator_loss = self.generator_loss(discriminator_net, discriminator_labels)


        recons_loss = tf.add(reco_loss_regre, reco_loss_cls)
        gan_loss = tf.add(discriminator_loss, generator_loss)
        all_loss = tf.add(recons_loss, gan_loss)

        # train net operator
        # train the network to minimize reconstruction loss and generator loss
        train_net_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
            .minimize(recons_loss + generator_loss, var_list=tf.compat.v1.get_collection('weight'))

        # train the latent space data to minimize reconstruction loss and classification loss
        train_hn_op = tf.compat.v1.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=h_update[0])

        # adjust the latent space data
        adj_hn_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
            .minimize(recons_loss, var_list=h_update[1])
        
        # generator
        #train_generator_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
        #    .minimize(generator_loss, var_list=tf.compat.v1.get_collection('weight'))
        # discriminator
        train_discriminator_op = tf.compat.v1.train.AdamOptimizer(learning_rate[0]) \
            .minimize(discriminator_loss, var_list=tf.compat.v1.get_collection('weight_discri'))


        return [train_net_op,
                train_hn_op, adj_hn_op,
                train_discriminator_op], \
               [all_loss, recons_loss, gan_loss,
                generator_loss, discriminator_loss,
                reco_loss_regre, reco_loss_cls]

        #return [train_net_op], \
        #       [reco_loss_regre]


    def transform_cat_outputs(self, net, vnum):
        out = tf.compat.v1.identity(net)
        ind = tf.where(self.cat_indicator[str(vnum)].astype('bool'))
        catT = tf.sigmoid(tf.gather(tf.transpose(out), ind))
        out = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(out), ind, catT))
        return out


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
        layer = tf.matmul(h, weight['w0']) + weight['b0']
        for num in range(1, len(self.layer_size[v])):
            layer = tf.nn.relu(layer)
            layer = tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)]
        return layer

    def Discriminator_net(self, x_real, x_generated, v):
        # concate and suffle data
        x_real = x_real * self.sn[str(v)] + x_generated * (1 - self.sn[str(v)])
        x_feat = tf.concat((x_real, x_generated), axis=0)
        y = tf.concat((tf.ones_like(self.gt), tf.zeros_like(self.gt)), axis=0)
        #sess = tf.compat.v1.InteractiveSession()
        #x_y = tf.random.shuffle(tf.concat((x_feat, tf.cast(y, tf.float32)), axis=1).eval())
        #x_feat = x_y[:, 0:-1]
        #y = tf.cast(x_y[:, -1], tf.bool)[:, None]
        weight_d = self.initialize_weight_for_discr(self.layer_size_d[v])
        layer_d = tf.matmul(x_feat, weight_d['w0']) + weight_d['b0']
        for num in range(1, len(self.layer_size_d[v])-1):
            layer = tf.nn.relu(layer_d)
            layer = tf.matmul(layer, weight_d['w' + str(num)]) + weight_d['b' + str(num)]
        return layer, y


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
            #all_weight['w0'] = tf.Variable(xavier_init(dims_net_discr[0], dims_net_discr[1]))
            #all_weight['b0'] = tf.Variable(tf.zeros([dims_net_discr[1]]))
            #tf.compat.v1.add_to_collection("weight_discri", all_weight['w' + str(0)])
            #tf.compat.v1.add_to_collection("weight_discri", all_weight['b' + str(0)])
            for num in range(0, len(dims_net_discr)):
                all_weight['w' + str(num-1)] = tf.Variable(xavier_init(dims_net_discr[num - 1], dims_net_discr[num]))
                all_weight['b' + str(num-1)] = tf.Variable(tf.zeros([dims_net_discr[num]]))
                tf.compat.v1.add_to_collection("weight_discri", all_weight['w' + str(num-1)])
                tf.compat.v1.add_to_collection("weight_discri", all_weight['b' + str(num-1)])
        return all_weight


    def reconstruction_loss(self, net):
        loss_regr = 0
        loss_cls = 0
        for num in range(self.view_num):
            ca_mask = tf.cast(tf.logical_not(self.cat_indicator[str(num)]), tf.float32)
            #loss_from_numeric_vs = tf.reduce_sum(
                #tf.boolean_mask(tf.multiply(tf.pow(tf.subtract(net[str(num)], self.input[str(num)]),
                #                                    2.0), ca_mask), self.sn[str(num)]), )
            loss_from_numeric_vs = tf.reduce_sum(
            tf.boolean_mask(tf.pow(tf.subtract(net[str(num)], self.input[str(num)]),
                                               2.0), self.sn[str(num)]),)
            loss_regr += loss_from_numeric_vs
            '''
            if self.cat_indicator[str(num)].sum() > 0:
                loss_from_cat_vs = 0.0
                for ith_cat in np.where(self.cat_indicator[str(num)])[0]:
                    logits = tf.gather(tf.transpose(net[str(num)]), ith_cat)
                    labels = tf.gather(tf.transpose(self.input[str(num)]), ith_cat)
                    sn = tf.gather(tf.transpose(self.sn[str(num)]), ith_cat)
                    loss_from_cat_vs += tf.compat.v1.losses.sigmoid_cross_entropy(
                            logits=logits*sn,
                            multi_class_labels=labels*sn)
                loss_cls += loss_from_cat_vs
            '''
        return loss_regr, loss_cls
    '''
    def classification_loss(self):
        F_h_h = tf.matmul(self.h_temp, tf.transpose(self.h_temp))
        F_hn_hn = tf.compat.v1.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.compat.v1.matrix_diag(F_hn_hn))
        classes = tf.reduce_max(self.gt) - tf.reduce_min(self.gt) + 1
        label_onehot = tf.one_hot(self.gt - 1, classes)  # gt begin from 1
        label_num = tf.compat.v1.reduce_sum(label_onehot, 0, keepdims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.compat.v1.tile(label_num, [self.trainLen, 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        gt_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.compat.v1.reduce_max(F_h_h_mean, axis=1, keepdims=False)
        theta = tf.cast(tf.not_equal(self.gt, gt_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.compat.v1.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean')
        return tf.compat.v1.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))
    '''

    def discriminator_loss(self,
            discriminator_outputs,
            discriminator_labels,
            label_smoothing=0,
            weights=1.0,
            scope=None,
            loss_collection=tf.compat.v1.GraphKeys.LOSSES,
            reduction=tf.compat.v1.losses.Reduction.SUM,
            add_summaries=False):
        loss = 0
        with tf.compat.v1.name_scope(scope, 'discriminator_loss',) as scope:
            # (discriminator_outputs, discriminator_labels, weights, label_smoothing)
            # -log((1 - label_smoothing) - sigmoid(D(x)))
            for ith_view in range(int(self.view_num)):
                loss += tf.compat.v1.losses.sigmoid_cross_entropy(
                    discriminator_labels[str(ith_view)],
                    discriminator_outputs[str(ith_view)],
                    weights,
                    label_smoothing,
                    scope,
                    loss_collection=None,
                    reduction=reduction)

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
            reduction=tf.compat.v1.losses.Reduction.SUM,
            add_summaries=False):
        loss = 0
        with tf.compat.v1.name_scope(scope, 'generator_loss') as scope:
            loss -= self.discriminator_loss(
                discriminator_outputs,
                discriminator_labels,
                weights=weights,  scope=scope,
                loss_collection=loss_collection, reduction=reduction)

        if add_summaries:
            tf.compat.v1.summary.scalar('generator_loss', loss)

        return loss



    def train(self, data, sn, gt, epoch, step=[5, 5, 5, 5]):
        global ReconstructionLoss, ClsLoss, GeneratorLoss, DiscriminatorLoss, AllLoss
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt[index]
        for i in sn.keys():
            sn[i] = sn[i][index]
        feed_dict = {self.input[str(v_num)]: data[str(v_num)][index] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[str(i)] for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        for iter in range(epoch):
            # update the network
            for i in range(step[0]):
                _, ReconstructionLoss, GeneratorLoss = self.sess.run(
                    [self.train_op[0], self.loss[1], self.loss[3]], feed_dict=feed_dict)

            # update the h
            for i in range(step[1]):
                _, AllLoss = self.sess.run(
                    [self.train_op[1], self.loss[0]], feed_dict=feed_dict)
            
            # update the generator
            #for i in range(step[3]):
            #    _, GeneratorLOSS = self.sess.run(
            #        [self.train_op[3], self.loss[3]], feed_dict=feed_dict)
            # updata the discriminator
            for i in range(step[3]):
                _, DiscriminatorLoss = self.sess.run(
                    [self.train_op[3], self.loss[4]], feed_dict=feed_dict)

            output = "Epoch : {:.0f}  ===> " \
                     "All Loss Loss = {:.4f}, " \
                     "Reconstruction Loss = {:.4f}, " \
                     "Generator Loss = {:.4f}, " \
                     "Discriminator Loss = {:.4f}" \
                .format((iter + 1),
                        AllLoss,
                        ReconstructionLoss,
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

    #def recover(self):
    #    net = dict()
    #    net= self.sess.run(self.output)
    #    return net


