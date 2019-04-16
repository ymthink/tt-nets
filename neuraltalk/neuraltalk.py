# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/12 17:01

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, LSTM, Add
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import Callback
from preprocess_data import *
from hyper_parameters import HyperParameters as hp
import keras


class ValidationCallback(Callback):

    def __init__(self, feats, id_list, idx2word, descriptions):
        super(Callback).__init__()
        self.feats = feats
        self.id_list = id_list
        self.idx2word = idx2word
        self.descriptions = descriptions

    def on_epoch_end(self, epoch, logs=None):
        n_samples = len(self.id_list)
        idx = np.random.randint(n_samples)
        image_id = self.id_list[idx]
        x1 = self.feats[idx-1:idx]
        x2 = np.zeros([1, hp.max_len])
        sent = ''
        y_idx = 0
        for i in range(hp.max_len):
            x2[0, i] = y_idx
            y_logit = self.model.predict([x1, x2])
            y_idx = np.argmax(y_logit)
            word = self.idx2word[y_idx]
            if word == '<E>':
                break
            sent += word + ' '
        print('INFO: ON_EPOCH_END')
        print('image_id:', image_id)
        print('ground truth:', self.descriptions[image_id])
        print('model output:', sent)


class NeuralTalk():
    def __init__(self):
        self.word2idx, self.idx2word, self.descriptions = load_vocab()
        self.vocab_size = len(self.word2idx)
        self.model = self.build_model(self.vocab_size)

    def train(self):
        train_id_list, train_X2_list = load_data(hp.train_images_file, self.word2idx, self.descriptions)
        test_id_list, test_X2_list = load_data(hp.test_images_file, self.word2idx, self.descriptions)
        train_feats = np.load(hp.train_feats_file)
        test_feats = np.load(hp.test_feats_file)
        print(test_feats[0].shape)
        optimizer = optimizers.Adam(lr=1e-3)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()
        cnt = 0
        for x2 in train_X2_list:
            cnt += len(x2)
        steps_per_epoch = cnt // hp.batch_size
        callback = ValidationCallback(test_feats, test_id_list, self.idx2word, self.descriptions)
        self.model.fit_generator(generator=self.generator(X2_list=train_X2_list, vocab_size=self.vocab_size, feats=train_feats),
                            steps_per_epoch=steps_per_epoch,
                            epochs=hp.n_epochs,
                            callbacks=[callback])

    @staticmethod
    def generator(X2_list, vocab_size, feats):
        count = 0
        X1, X2, Y = [], [], []
        while True:
            for i in range(len(X2_list)):
                for seq in X2_list[i]:
                    for j in range(len(seq)):
                        inp_seq, target = seq[:j], seq[j]
                        x2 = np.lib.pad(inp_seq, [0, hp.max_len-len(inp_seq)], 'constant', constant_values=(0, 0))
                        X1.append(feats[i])
                        X2.append(x2)
                        y = to_categorical([target], num_classes=vocab_size)[0]
                        Y.append(y)
                        count += 1
                        if count == hp.batch_size:
                            yield np.array([[np.array(X1), np.array(X2)], np.array(Y)])
                            count = 0
                            X1.clear()
                            X2.clear()
                            Y.clear()

    @staticmethod
    def build_model(vocab_size):
        img_inp = Input(shape=(2048,))
        img_embedding = Dense(hp.embedding_dim, activation='relu')(img_inp)

        txt_inp = Input(shape=(hp.max_len,))
        txt_embedding = Embedding(vocab_size, hp.embedding_dim, mask_zero=True)(txt_inp)
        txt_lstm = LSTM(256)(txt_embedding)

        merged_inp = keras.layers.add([img_embedding, txt_lstm])
        decoder = Dense(256, activation='relu')(merged_inp)
        outp = Dense(vocab_size, activation='softmax')(decoder)

        inputs = [img_inp, txt_inp]
        model = Model(inputs=inputs, outputs=outp)
        return model


if __name__ == '__main__':
    nt = NeuralTalk()
    nt.train()





