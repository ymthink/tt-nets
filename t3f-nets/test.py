# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/2 14:36


import numpy as np
import tensorflow as tf
import t3f
import sys

sess = tf.Session()

# w_mat = np.random.normal(0, 0.5, [7, 6, 8, 5])
# w_tt = t3f.to_tt_tensor(w_mat)
# w_ = t3f.full(w_tt)
#
# err = tf.abs(tf.reduce_sum(w_mat - w_))
#
# sess.run(tf.global_variables_initializer())
# print(sess.run(err))

w_mat = np.random.normal(0, 0.5, [5, 5, 64, 128])
w = tf.Variable(initial_value=w_mat)
h1 = tf.reshape(w_mat, [5, 5, 8, 8, 8, 4, 4])
h2 = tf.reshape(h1, [5, 5, 64, 128])
err = tf.abs(tf.reduce_sum(h2-w))
sess.run(tf.global_variables_initializer())
print(sess.run(err))


