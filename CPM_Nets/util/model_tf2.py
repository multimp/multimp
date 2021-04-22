import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
#tf.disable_eager_execution()

class CPMNets():
    """build model
    """
    def __init__(self, view_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        self.learning_rate = learning_rate
        # initialize network
        self.decoders = Decoders(self.view_num, self.layer_size, self.lsd_dim)

    def bulid_model(self, h_update, learning_rate, losses):
        # train net operator
        # train the network to minimize reconstruction loss
        train_net_op = tf.keras.optimizers.Adam(learning_rate[0]).minimize(loss=losses[0], var_list=self.decoders.trainable_weights)
        # train the latent space data to minimize reconstruction loss and classification loss
        train_hn_op = tf.keras.optimizers.Adam(learning_rate[1]).minimize(loss=losses[1], var_list=h_update[0])
        # adjust the latent space data
        adj_hn_op = tf.keras.optimizers.Adam(learning_rate[0]).minimize(loss=losses[2], var_list=h_update[1])

        return [train_net_op, train_hn_op, adj_hn_op]



    def reconstruction_loss(self, recons, input,  sn):
        loss = 0
        for num in range(self.view_num):
            loss = loss + tf.keras.backend.sum(
                tf.keras.backend.pow(recons[str(num)] - input[str(num)], 2.0) * sn[str(num)]
            )
        return loss

    def classification_loss(self, h_temp, gt):
        F_h_h = tf.matmul(h_temp, tf.transpose(h_temp))
        F_hn_hn = tf.compat.v1.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.compat.v1.matrix_diag(F_hn_hn))
        classes = tf.reduce_max(gt) - tf.reduce_min(gt) + 1
        label_onehot = tf.one_hot(gt - 1, classes)  # gt begin from 1
        label_num = tf.compat.v1.reduce_sum(label_onehot, 0, keepdims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.compat.v1.tile(label_num, [self.trainLen, 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        gt_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.compat.v1.reduce_max(F_h_h_mean, axis=1, keepdims=False)
        theta = tf.cast(tf.not_equal(gt, gt_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.compat.v1.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean')
        return tf.compat.v1.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))

    def train(self, data, sn, gt, epochs, step=[5, 5]):
        global Reconstruction_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt[index]
        sn = sn[index]
        h_index = index.reshape((self.trainLen, 1))
        sn_dict = {}
        input_dict = {}
        for i in range(self.view_num):
            sn_dict[str(i)] = sn[:, i].reshape(self.trainLen, 1)
        for v_num in range(self.view_num):
            input_dict[str(v_num)] = tf.convert_to_tensor(data[str(v_num)][index], dtype=tf.float32)
        '''
        feed_dict = {self.input[str(v_num)]: data[str(v_num)][index] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.trainLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        '''
        self.latent_space = LatentSpace(self.view_num, self.trainLen, self.testLen, self.layer_size, self.lsd_dim, h_index)
        train_net_op = tf.keras.optimizers.Adam(self.learning_rate[0])
        train_hn_op = tf.keras.optimizers.Adam(self.learning_rate[1])
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for i in range(step[0]):

                with tf.GradientTape(persistent=True) as tape:
                    recons = self.decoders(tf.convert_to_tensor(self.latent_space.h_temp))
                    reco_loss = self.reconstruction_loss(recons, input_dict, sn_dict)
                vars = self.decoders.trainable_weights
                grads = tape.gradient(reco_loss, vars)
                # Ask the optimizer to apply the processed gradients.
                train_net_op.apply_gradients(zip(grads, vars))
                #train_net_op.minimize(loss=lambda: reco_loss, var_list=lambda: self.decoders.trainable_weights)

            # update the h
            for i in range(step[1]):
                with tf.GradientTape(persistent=True) as tape:
                    recons = self.latent_space(self.decoders.nets)
                    reco_loss = self.reconstruction_loss(recons, input_dict, sn_dict)
                vars = self.latent_space.trainable_weights
                grads = tape.gradient(reco_loss, vars)
                # Ask the optimizer to apply the processed gradients.
                train_hn_op.apply_gradients(zip(grads, vars))

            output = "Epoch : {:.0f}  ===> Recons Loss = {:.4f} ".format((iter + 1), loss_value)
            print(output)

    def test(self, data, sn, gt, epoch, steps=5):
        '''
        feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.testLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index:
                              np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})
        '''
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt[index]
        sn = sn[index]
        h_index = index.reshape((self.trainLen, 1))
        input_dict = {}
        sn_dict = {}
        for i in range(self.view_num):
            sn_dict[str(i)] = sn[:, i].reshape(self.trainLen, 1)
        for v_num in range(self.view_num):
            input_dict[str(v_num)] = data[str(v_num)][index]

        for iter in range(epoch):
            # update the h
            for i in range(steps):
                with tf.GradientTape() as tape:
                    recons = self.latent_space(self.decoders, h_index, training=True)
                    reco_loss = self.reconstruction_loss(input_dict, recons, sn_dict)
                    #class_loss = self.classification_loss(self.latent_space.h_temp, gt)
                    #all_loss = tf.add(reco_loss, self.lamb * class_loss)
                grads = tape.gradient(reco_loss, self.latent_space.h_test_update)
                self.train_op[2].apply_gradients(zip(grads, self.latent_space.h_test_update))

            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" .format((iter + 1), Reconstruction_LOSS)
            print(output)






class Decoders(tf.keras.Model):
    def __init__(self, view_num, layer_size, lsd_dim):
        super(Decoders, self).__init__()
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.nets = dict()
        for v_num in range(self.view_num):
            self.nets[str(v_num)] = self.Encoding_net(v_num)

    def Encoding_net(self, v):
        current_decoder = self.initialize_decoder_branch(self.layer_size[v])
        return current_decoder

    def initialize_decoder_branch(self, dims_net):
        '''
        current_branch = tf.keras.models.Sequential()
        current_branch.add(tf.keras.Input(shape=(self.lsd_dim,)))
        current_branch.add(tf.keras.layers.Dense(units=dims_net[0],
                              bias_initializer=tf.keras.initializers.zeros,
                              kernel_initializer=tf.keras.initializers.glorot_uniform))

        for num in range(1, len(dims_net)):
            current_branch.add(tf.keras.activations.relu())
            current_branch.add(tf.keras.layers.Dropout(0.9))
            current_branch.add(tf.keras.layers.Dense(units=dims_net[num],
                                                     bias_initializer=tf.keras.initializers.zeros,
                                                     kernel_initializer=tf.keras.initializers.he_normal))
                                                     '''
        all_weight = dict()
        all_weight['w0'] = tf.Variable(xavier_init(self.lsd_dim, dims_net[0]))
        all_weight['b0'] = tf.Variable(tf.zeros([dims_net[0]]))
        for num in range(1, len(dims_net)):
            all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
            all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net[num]]))

        return all_weight

    def call(self, h):
        #h_temp = tf.gather_nd(self.h, h_index)
        layers = dict()
        for i_key in self.nets.keys():
            #layer = self.nets[i_key](h)
            #layers[i_key] = layer
            layer = tf.matmul(h, self.nets[i_key]['w0']) + self.nets[i_key]['b0']
            layers[i_key] = layer
        return layers


class LatentSpace(tf.keras.Model):
    def __init__(self, view_num, trainLen, testLen, layer_size, lsd_dim, h_index):
        super(LatentSpace, self).__init__()
        self.view_num = view_num
        self.trainLen = trainLen
        self.testLen = testLen
        self.lsd_dim = lsd_dim
        self.layer_size = layer_size
        self.h_index = h_index
        # initialize latent space data
        self.h_train, self.h_train_update = self.H_init('train')
        self.h_test, self.h_test_update = self.H_init('test')
        self.h = tf.concat([self.h_train, self.h_test], axis=0)
        self.h_temp = tf.Variable(tf.gather_nd(self.h, h_index), trainable=True)

    def H_init(self, a):
        if a == 'train':
            h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim), trainable=True)
            #h_update = self.trainable_variables()
            return h, h
        elif a == 'test':
            h = tf.Variable(xavier_init(self.testLen, self.lsd_dim), trainable=True)
            #h_update = self.trainable_variables()
            return h, h

    def call(self, decoders):
        layers = dict()
        for i_key in decoders.keys():
            layer = tf.matmul(self.h_temp, decoders[i_key]['w0']) + decoders[i_key]['b0']
            layers[i_key] = layer
        return layers