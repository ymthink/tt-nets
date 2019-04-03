# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/2 20:12


import numpy as np
import tensorflow as tf
import t3f


def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / np.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)


def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size ** 2
    fan_out = output_dim * filter_size ** 2 / (stride ** 2)
    stddev = np.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)


def tt_he_init(filter_size, in_ch_modes, out_ch_modes, stride):
    input_dim = np.prod(in_ch_modes)
    output_dim = np.prod(out_ch_modes)

    fan_in = input_dim * filter_size ** 2
    fan_out = output_dim * filter_size ** 2 / (stride ** 2)
    stddev = np.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    tensor_shape = [filter_size, filter_size] + in_ch_modes + out_ch_modes
    return tf.random_uniform(shape=tensor_shape, minval=minval, maxval=maxval)


def conv2d(inp, out_ch, filter_size, stride, layer_id, padding='SAME'):
    in_shape = inp.get_shape().as_list()
    init_w = he_init([filter_size, filter_size, in_shape[-1], out_ch], stride)
    weights = tf.get_variable(
        'W' + str(layer_id),
        initializer=init_w
    )
    init_b = tf.zeros([out_ch])
    bias = tf.get_variable(
        'b' + str(layer_id),
        initializer=init_b
    )

    outp = tf.add(tf.nn.conv2d(
        inp,
        weights,
        strides=[1, stride, stride, 1],
        padding=padding
    ), bias)

    return outp


def ttconv2d(inp, in_ch_modes, out_ch_modes, tt_rank, filter_size, stride, layer_id, init_w=None, padding='SAME'):
    in_ch = np.prod(in_ch_modes)
    out_ch = np.prod(out_ch_modes)
    tensor_shape = [filter_size, filter_size] + in_ch_modes + out_ch_modes
    if init_w is None:
        init_w = tt_he_init(filter_size, in_ch_modes, out_ch_modes, stride)
    else:
        init_w = tf.reshape(init_w, shape=tensor_shape)
    weights = t3f.to_tt_tensor(init_w, max_tt_rank=tt_rank)
    filters = tf.reshape(t3f.full(weights), [filter_size, filter_size, in_ch, out_ch])
    init_b = tf.zeros(shape=[out_ch])
    bias = tf.get_variable(
        'b' + str(layer_id),
        initializer=init_b
    )

    outp = tf.add(tf.nn.conv2d(
        inp,
        filters,
        strides=[1, stride, stride, 1],
        padding=padding
    ), bias)

    return outp

def dense(inp, out_dim, layer_id):
    in_dim = np.prod(inp.get_shape().as_list()[1:])
    init_w = xavier_init([in_dim, out_dim])
    weights = tf.get_variable('W' + str(layer_id), initializer=init_w)
    init_b = tf.zeros([out_dim])
    bias = tf.get_variable('b' + str(layer_id), initializer=init_b)
    h = tf.reshape(inp, [-1, in_dim])

    outp = tf.add(tf.matmul(h, weights), bias)

    return outp


def batch_norm(inp, scale=False):
    outp = tf.contrib.layers.batch_norm(inp, scale=scale)
    return outp