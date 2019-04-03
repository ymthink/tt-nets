# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/1 13:36

import numpy as np
import tensorflow as tf
import t3f
from keras.datasets import cifar10
from keras.utils import to_categorical
import sys
sys.path.append('../')
from layers import *


class TTConvNet:
    def __init__(self, img_shape, num_class):
        self.images = tf.placeholder(dtype=tf.float32, shape=(None, img_shape[0], img_shape[1], img_shape[2]))
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, ))
        h = conv2d(self.images, out_ch=64, filter_size=5, stride=1, layer_id=0)
        h = batch_norm(h)
        h = tf.nn.relu(h)
        h = ttconv2d(
            h,
            in_ch_modes=[4, 4, 4],
            out_ch_modes=[4, 8, 4],
            tt_rank=21,
            filter_size=5,
            stride=1,
            layer_id=1
        )
        h = batch_norm(h)
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        h = ttconv2d(
            h,
            in_ch_modes=[4, 8, 4],
            out_ch_modes=[4, 8, 4],
            tt_rank=23,
            filter_size=3,
            stride=1,
            layer_id=2
        )
        h = batch_norm(h)
        h = tf.nn.relu(h)
        h = tf.nn.avg_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        h = dense(h, num_class, layer_id=3)
        self.logits = h

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        correct_flags = tf.nn.in_top_k(self.logits, self.labels, 1)
        self.eval = tf.cast(correct_flags, tf.int32)

    def train(self, epochs=10, batch_size=32, lr=0.001):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train / 127.5 - 1.0
        x_test = x_test / 127.5 - 1.0

        # y_train = to_categorical(y_train, num_classes=10)
        # y_test = to_categorical(y_test, num_classes=10)
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
        for epoch in range(epochs):
            for i in range(n_step):
                img_samples = x_train[i*batch_size:(i+1)*batch_size]
                label_samples = y_train[i*batch_size:(i+1)*batch_size]
                _, loss = sess.run([opt, self.loss], feed_dict={self.images:img_samples, self.labels:label_samples})
                if i % 100 == 0:
                    print('step {} / {},'.format(i, n_step),'loss:', loss)
            eval = sess.run(self.eval, feed_dict={self.images:x_test[0:200], self.labels:y_test[0:200]})
            print('evaluation acc:', np.sum(eval) / 200)


if __name__ == '__main__':
    net = TTConvNet(img_shape=(32, 32, 3), num_class=10)
    net.train()





