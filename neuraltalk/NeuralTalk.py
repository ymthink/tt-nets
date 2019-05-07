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
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import keras
import sys

sys.path.append('../')
from neuraltalk.preprocess_data import *
from neuraltalk.hyper_parameters import HyperParameters as hp
from TTLAYERS.TTRNN import *
from TTLAYERS.TTFC import *


class ValidationCallback(Callback):

    def __init__(self, feats, id_list, idx2word, word2idx, descriptions, is_TT):
        super(Callback).__init__()
        self.feats = feats
        self.id_list = id_list
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.descriptions = descriptions
        self.is_TT = is_TT

    def on_epoch_end(self, epoch, logs=None):
        n_samples = len(self.id_list)
        idx = np.random.randint(n_samples)
        image_id = self.id_list[idx]
        x1 = self.feats[idx:(idx+1)]
        x2 = np.zeros([1, hp.max_len])
        sent = ''
        y_idx = self.word2idx['<S>']
        for i in range(hp.max_len):
            x2[0, i] = y_idx
            y_logit = self.model.predict([x1, x2])
            try:
                y_idx = np.argmax(y_logit)
            except:
                print('ERROR: y_logit error!')
                print('idx:', idx)
                print('features:', self.feats[idx])
                print('y_logit:', y_logit)
                return
            word = self.idx2word[y_idx]
            if word == '<E>':
                break
            sent += word + ' '
        print('INFO: ON_EPOCH_END')
        print('image_id:', image_id)
        print('ground truth:', self.descriptions[image_id])
        print('model output:', sent)
        if self.is_TT:
            self.model.save_weights('model_weights_tt.h5')
        else:
            self.model.save_weights('model_weights_base.h5')
        print('INFO: model_weights have been saved.')


class BleuCallback(Callback):

    def __init__(self, feats, id_list, idx2word, word2idx, descriptions, is_TT):
        super(Callback).__init__()
        self.feats = feats
        self.id_list = id_list
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.descriptions = descriptions
        self.is_TT = is_TT

    def on_epoch_end(self, epoch, logs=None):
        print('INFO: ON_EPOCH_END begin to calculate current BLEU score.')
        score_list = []
        for idx in range(len(self.id_list)):
            image_id = self.id_list[idx]
            references = self.descriptions[image_id]
            candidate = []
            x1 = self.feats[idx:idx+1]
            x2 = np.zeros([1, hp.max_len])
            y_idx = self.word2idx['<S>']
            for i in range(hp.max_len):
                x2[0, i] = y_idx
                y_logit = self.model.predict([x1, x2])
                y_idx = np.argmax(y_logit)
                word = self.idx2word[y_idx]
                if word == '<E>':
                    break
                candidate.append(word)

            score = sentence_bleu(references, candidate, weights=[1, 0, 0, 0])
            score_list.append(score)

        print('BLEU-1: {:.4f}'.format(np.mean(score_list)))

        if self.is_TT:
            self.model.save_weights('model_weights_tt.h5')
        else:
            self.model.save_weights('model_weights_base.h5')
        print('INFO: model_weights have been saved.')


class NeuralTalk():
    def __init__(self, is_TT=False):
        self.is_TT = is_TT
        self.word2idx, self.idx2word, self.descriptions = load_vocab()
        self.vocab_size = len(self.word2idx)
        self.model = self.build_model(self.vocab_size, self.is_TT)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, validation=0):
        self.model.summary()

        train_id_list, train_X2_list = load_data(hp.train_images_file, self.word2idx, self.descriptions)
        test_id_list, test_X2_list = load_data(hp.test_images_file, self.word2idx, self.descriptions)
        train_feats = load_feats(hp.pkl_file, train_id_list)
        test_feats = load_feats(hp.pkl_file, test_id_list)

        steps_per_epoch = len(train_X2_list) // hp.train_size

        if validation == 0:
            callback = ValidationCallback(test_feats, test_id_list, self.idx2word, self.word2idx, self.descriptions, self.is_TT)
        elif validation == 1:
            callback = BleuCallback(test_feats, test_id_list, self.idx2word, self.word2idx, self.descriptions, self.is_TT)
        self.model.fit_generator(generator=self.generator(X2_list=train_X2_list, vocab_size=self.vocab_size, feats=train_feats),
                            steps_per_epoch=steps_per_epoch,
                            epochs=hp.n_epochs,
                            verbose=2,
                            callbacks=[callback])

    def eval(self):
        if self.is_TT:
            self.model.load_weights('model_weights_tt.h5')
        else:
            self.model.load_weights('model_weights_base.h5')
        test_id_list, test_X2_list = load_data(hp.test_images_file, self.word2idx, self.descriptions)
        test_feats = load_feats(hp.pkl_file, test_id_list)
        score_list = []
        for idx in range(len(test_id_list)):
            image_id = test_id_list[idx]
            references = self.descriptions[image_id]
            candidate = []
            x1 = test_feats[idx:idx+1]
            x2 = np.zeros([1, hp.max_len])
            y_idx = self.word2idx['<S>']
            for i in range(hp.max_len):
                x2[0, i] = y_idx
                y_logit = self.model.predict([x1, x2])
                y_idx = np.argmax(y_logit)
                word = self.idx2word[y_idx]
                if word == '<E>':
                    break
                candidate.append(word)
            score = sentence_bleu(references, candidate, weights=[1, 0, 0, 0])
            score_list.append(score)
        print('BLEU-1: {:.4f}'.format(np.mean(score_list)))


    @staticmethod
    def generator(X2_list, vocab_size, feats):
        X1, X2, Y = [], [], []
        count = 0
        while True:
            for i in range(len(X2_list)):
                seqs = X2_list[i]
                for seq in seqs:
                    for j in range(1, len(seq)):
                        inp_seq, target = seq[:j], seq[j]
                        x2 = np.lib.pad(inp_seq, [0, hp.max_len-len(inp_seq)], 'constant', constant_values=(0, 0))
                        X1.append(feats[i])
                        X2.append(x2)
                        y = to_categorical([target], num_classes=vocab_size)[0]
                        Y.append(y)
                count += 1
                if count == hp.train_size:
                    yield np.array([[np.array(X1), np.array(X2)], np.array(Y)])
                    count = 0
                    X1.clear()
                    X2.clear()
                    Y.clear()

    @staticmethod
    def build_model(vocab_size, is_TT=False):
        img_inp = Input(shape=(2048,))
        img_dropout = Dropout(0.5)(img_inp)
        if is_TT:
            img_embedding = TT_Dense(tt_input_shape=[4,8,8,8], tt_output_shape=[4,4,4,4], tt_ranks=[1,3,3,3,1], activation='relu', use_bias=True)(img_dropout)
        else:
            img_embedding = Dense(hp.embedding_dim, activation='relu')(img_dropout)

        txt_inp = Input(shape=(hp.max_len,))
        txt_embedding = Embedding(vocab_size, hp.embedding_dim, mask_zero=True)(txt_inp)
        txt_dropout = Dropout(0.5)(txt_embedding)
        if is_TT:
            txt_lstm = TT_LSTM(tt_input_shape=[4,4,4,4], tt_output_shape=[4,4,4,4], tt_ranks=[1,3,3,3,1], use_bias=True)(txt_dropout)
        else:
            txt_lstm = LSTM(256)(txt_dropout)

        merged_inp = keras.layers.add([img_embedding, txt_lstm])
        decoder = Dense(256, activation='relu')(merged_inp)
        outp = Dense(vocab_size, activation='softmax')(decoder)

        inputs = [img_inp, txt_inp]
        model = Model(inputs=inputs, outputs=outp)
        return model






