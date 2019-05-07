# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/15 15:13

import string
import numpy as np
import pickle
import os
from keras.preprocessing.image import load_img, img_to_array
from collections import Counter
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
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
        # image_desc = [word for word in image_desc if len(word) > 1]
        image_desc = [word for word in image_desc if word.isalpha()]
        # desc = ' '.join(image_desc)
        desc = image_desc
        descriptions[image_id].append(desc)
        vocab_count.update(image_desc)

    fin.close()

    vocab = ['<PAD>', '<UNKNOWN>', '<S>', '<E>']
    vocab.extend([word for word in vocab_count.keys() if vocab_count[word] >= hp.min_count])

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
            x = []
            x.append(word2idx['<S>'])
            x.extend([word2idx.get(word, 1) for word in desc])
            x.append(word2idx['<E>'])
            if len(x) <= hp.max_len:
                sents.append(x)

        if len(sents) > 0:
            X2_list.append(sents)
            id_list.append(image_id)

    return id_list, X2_list


def load_feats(filename, id_list):
    all_features = pickle.load(open(filename, 'rb'))
    feats = np.array([all_features[id] for id in id_list])
    return feats


def save_image_feats():
    inception_v3 = InceptionV3(weights='imagenet')
    model = Model(inception_v3.input, inception_v3.layers[-2].output)
    img_files = os.listdir(hp.dataset_path)
    feats_dict = dict()
    for img_file in img_files:
        if img_file.endswith('.jpg'):
            img_id = img_file.split('.')[0]
            img_filepath = hp.dataset_path + img_file
            img = load_img(img_filepath, target_size=(299, 299))
            img = img_to_array(img)
            x = preprocess_input(img)
            x = np.expand_dims(x, axis=0)
            feat = model.predict(x)
            feat = feat.flatten()
            feats_dict.update({img_id:feat})
    with open(hp.pkl_file, 'wb') as handle:
        pickle.dump(feats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    save_image_feats()









