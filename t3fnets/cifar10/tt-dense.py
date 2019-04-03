# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/3/31 15:42

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras import optimizers
import numpy as np
import tensorflow as tf
import t3f
import sys

__console__ = sys.stdout
fout = open('./tt-dense.txt', 'w')

num_epoch = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 127.5 - 1.0
x_test = x_test / 127.5 -1.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

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

sys.stdout = fout
model.summary()
sys.stdout = __console__

optimizer = optimizers.Adam(lr=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train, y_train,
          epochs=num_epoch,
          batch_size=32,
          validation_data=(x_test, y_test)
)
sys.stdout = fout
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss, 'acc:', acc)
fout.close()



