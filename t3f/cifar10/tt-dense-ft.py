# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/3/31 17:35

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras import optimizers
import keras.backend as K
import numpy as np
import tensorflow as tf
import t3f
import sys

num_epoch = 100

sess = tf.InteractiveSession()
K.set_session(sess)

__console__ = sys.stdout
fout = open('./finetuning.txt', 'w')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 127.5 - 1.0
x_test = x_test / 127.5 -1.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

sys.stdout = fout
model.summary()
sys.stdout = __console__
fout.flush()

optimizer = optimizers.Adam(lr=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train,
          epochs=num_epoch,
          batch_size=32,
          validation_data=(x_test, y_test),
          callbacks=[TensorBoard(log_dir='./logs/dense2tt-finetuning')])

sys.stdout = fout
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss, 'acc:', acc)
fout.flush()

W1 = model.trainable_weights[0]
b1 = model.get_weights()[1]
W2 = model.trainable_weights[2]
b2 = model.get_weights()[3]
other_params = model.get_weights()[4:]

W1_tt = t3f.to_tt_matrix(W1,
                         shape=[[4, 8, 4, 8, 3], [4, 4, 4, 4, 4]],
                         max_tt_rank=9)
W2_tt = t3f.to_tt_matrix(W2,
                         shape=[[4, 8, 4, 8], [4, 4, 4, 4]],
                         max_tt_rank=9)

tt_rank2 = W2_tt.get_tt_ranks()[1]
W1_tt_cores = sess.run(W1_tt.tt_cores)
W2_tt_cores = sess.run(W2_tt.tt_cores)

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
tt_layer1 = t3f.nn.KerasDense(
    input_dims=[4, 8, 4, 8, 3],
    output_dims=[4, 4, 4, 4, 4],
    tt_rank=9,
    activation='relu',
    bias_initializer=0
)
model.add(tt_layer1)
tt_layer2 = t3f.nn.KerasDense(
    input_dims=[4, 8, 4, 8],
    output_dims=[4, 4, 4, 4],
    tt_rank=9,
    activation='relu',
    bias_initializer=0
)
model.add(tt_layer2)
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

optimizer = optimizers.Adam(lr=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print('new acc:', model.evaluate(x_test, y_test)[1])

sys.stdout = __console__
vars = list()
vars.extend(W1_tt_cores)
vars.append(b1)
vars.extend(W2_tt_cores)
vars.append(b2)
vars.extend(other_params)
model.set_weights(vars)

sys.stdout = __console__
model.fit(x_train, y_train,
          epochs=num_epoch,
          batch_size=32,
          validation_data=(x_test, y_test)
)
sys.stdout = fout
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss, 'acc:', acc)
fout.close()



