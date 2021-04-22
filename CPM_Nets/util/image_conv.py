import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
tf.compat.v1.disable_eager_execution()
# MNIST 数据集参数
num_classes = 10 # 所有类别（数字 0-9）

# 训练参数
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# 网络参数
conv1_filters = 32 # 第一层卷积层卷积核的数目
conv2_filters = 64 # 第二层卷积层卷积核的数目
fc1_units = 1024 # 第一层全连接层神经元的数目


def conv2d(x, W, b, strides=1):
    # Conv2D包装器, 带有偏置和relu激活
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D包装器
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

random_normal = tf.initializers.RandomNormal()

weights = {
    # 第一层卷积层： 5 * 5卷积，1个输入， 32个卷积核(MNIST只有一个颜色通道)
    'wc1': tf.Variable(random_normal([5, 5, 1, conv1_filters])),
    # 第二层卷积层： 5 * 5卷积，32个输入， 64个卷积核
    'wc2': tf.Variable(random_normal([5, 5, conv1_filters, conv2_filters])),
    # 全连接层： 7*7*64 个输入， 1024个神经元
    'wd1': tf.Variable(random_normal([7*7*64, fc1_units])),
    # 全连接层输出层: 1024个输入， 10个神经元（所有类别数目)
    'out': tf.Variable(random_normal([fc1_units, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([conv1_filters])),
    'bc2': tf.Variable(tf.zeros([conv2_filters])),
    'bd1': tf.Variable(tf.zeros([fc1_units])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

def conv_net(x):
    # 输入形状：[-1, 28, 28, 1]。一批28*28*1（灰度）图像
    x = tf.reshape(x, [-1, 28, 28, 1])

    # 卷积层, 输出形状：[ -1, 28, 28 ,32]
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # 最大池化层（下采样） 输出形状：[ -1, 14, 14, 32]
    conv1 = maxpool2d(conv1, k=2)

    # 卷积层， 输出形状：[ -1, 14, 14, 64]
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # 最大池化层（下采样） 输出形状：[ -1, 7, 7, 64]
    conv2 = maxpool2d(conv2, k=2)

    # 修改conv2的输出以适应完全连接层的输入， 输出形状：[-1, 7*7*64]
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # 全连接层， 输出形状： [-1, 1024]
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
     # 将ReLU应用于fc1输出以获得非线性
    fc1 = tf.nn.relu(fc1)

    # 全连接层，输出形状 [ -1, 10]
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # 应用softmax将输出标准化为概率分布
    return tf.nn.softmax(out)