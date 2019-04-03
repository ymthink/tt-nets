# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/2 20:55


import numpy as np
import tensorflow as tf
import t3f
from keras.datasets import cifar10
import sys
sys.path.append('../../')
from t3fnets.layers import *
from t3fnets.net import *


class ConvNet(Net):
    def __init__(self, img_shape, num_class):
        super(Net, self).__init__()
        self.images = tf.placeholder(dtype=tf.float32, shape=(None, img_shape[0], img_shape[1], img_shape[2]))
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ))
        h = conv2d(self.images, out_ch=64, filter_size=5, stride=1, layer_id=0)
        h = batch_norm(h)
        h = tf.nn.relu(h)
        h = conv2d(h, out_ch=128, filter_size=5, stride=1, layer_id=1)
        h = batch_norm(h)
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        h = conv2d(h, out_ch=128, filter_size=3, stride=1, layer_id=2)
        h = batch_norm(h)
        h = tf.nn.relu(h)
        h = tf.nn.avg_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        h = dense(h, num_class, layer_id=3)
        # self.logits = tf.nn.softmax(h)
        self.logits = h

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        correct_flags = tf.nn.in_top_k(self.logits, self.labels, 1)
        self.pred = tf.cast(correct_flags, tf.int32)

    def train(self, epochs=10, batch_size=32, lr=1e-3):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train / 127.5 - 1.0
        x_test = x_test / 127.5 - 1.0

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        n_sample = len(x_train)
        n_step = n_sample // batch_size

        opt = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        for epoch in range(epochs):
            self.eval(self.images, self.labels, self.pred, sess, x_test, y_test, batch_size)
            for i in range(n_step):
                img_samples = x_train[i*batch_size:(i+1)*batch_size]
                label_samples = y_train[i*batch_size:(i+1)*batch_size]
                _, loss = sess.run([opt, self.loss], feed_dict={self.images:img_samples, self.labels:label_samples})
                if i % 100 == 0:
                    print('epoch {} / {}, step {} / {},'.format(epoch, epochs, i, n_step),'loss:', loss)
            W1 = sess.run('W1:0')
            W2 = sess.run('W2:0')
            np.save('W_init', [W1, W2])
            print('INFO: The weights of layer_1 and layer_2 have been saved.')
        self.eval(self.images, self.labels, self.pred, sess, x_test, y_test, batch_size)


if __name__ == '__main__':
    net = ConvNet(img_shape=(32, 32, 3), num_class=10)
    net.train()
