# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/3 13:23


import numpy as np
import tensorflow as tf
import sys

from .layers import *


class Net(object):
    def __init__(self, ):
        pass

    def eval(self, images, labels, pred, sess, x_test, y_test, batch_size):
        n_sample = len(x_test)
        n_step = n_sample // batch_size
        sum_pred = 0
        for i in range(n_step):
            img_samples = x_test[i * batch_size:(i + 1) * batch_size]
            label_samples = y_test[i * batch_size:(i + 1) * batch_size]
            cur_pred = np.sum(sess.run(pred, feed_dict={images:img_samples, labels:label_samples}))
            sum_pred += cur_pred
        print('Accuracy: {:.5f}'.format(sum_pred / (n_step * batch_size)))