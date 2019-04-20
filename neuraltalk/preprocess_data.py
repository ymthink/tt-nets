# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/15 15:13

import string
import numpy as np
import keras
from collections import Counter
from keras.applications.inception_v3 import  InceptionV3
from keras.models import Model
from hyper_parameters import HyperParameters as hp


def load_vocab():
    fin = open(hp.token_file, 'r')
    table = str.maketrans('', '', string.punctuation)
    descriptions = dict()
    vocab_count = Counter()

    for line in fin.readlines():
        splits = line.split()
        image_name = splits[0]
        image_desc = splits[1:]
        image_id = image_name.split('.')[0]
        if image_id not in descriptions.keys():
            descriptions[image_id] = list()
        image_desc = [word.lower() for word in image_desc]
        image_desc = [w.translate(table) for w in image_desc]
        image_desc = [word for word in image_desc if len(word) > 1]
        image_desc = [word for word in image_desc if word.isalpha()]
        # desc = ' '.join(image_desc)
        desc = image_desc
        descriptions[image_id].append(desc)
        vocab_count.update(image_desc)

    fin.close()

    vocab = ['<PAD>', '<UNKNOWN>', '<S>', '<E>']
    vocab.extend([word for word in vocab_count.keys() if vocab_count[word] > hp.min_count])

    word2idx = {word : idx for idx, word in enumerate(vocab)}
    idx2word = {idx : word for idx, word in enumerate(vocab)}

    return word2idx, idx2word, descriptions


def load_data(filename, word2idx, descriptions):
    fin = open(filename)
    id_list = []
    X2_list = []
    for line in fin.readlines():
        image_id = line.split('.')[0]
        sents = []
        for desc in descriptions[image_id]:
            x = [word2idx.get(word, 1) for word in desc]
            x.append(word2idx['<E>'])
            if len(x) <= hp.max_len:
                sents.append(x)

        if len(sents) > 0:
            X2_list.append(sents)
            id_list.append(image_id)

    return id_list, X2_list


def save_image_feats(id_list, filename):
    inception_v3 = InceptionV3(weights='imagenet')
    model = Model(inception_v3.input, inception_v3.layers[-2].output)
    feat_list = []
    for id in id_list:
        image_filepath = hp.dataset_path + id + '.jpg'
        img = keras.preprocessing.image.load_img(image_filepath, target_size=(299, 299))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        feat = model.predict(img)
        feat = np.squeeze(feat)
        feat_list.append(feat)

    feats = np.array(feat_list)
    np.save(filename, feats)


if __name__ == '__main__':
    word2idx, idx2word, descriptions = load_vocab()
    train_id_list, _ = load_data(hp.train_images_file, word2idx, descriptions)
    save_image_feats(train_id_list, hp.train_feats_file)
    test_id_list, _ = load_data(hp.test_images_file, word2idx, descriptions)
    save_image_feats(test_id_list, hp.test_feats_file)









