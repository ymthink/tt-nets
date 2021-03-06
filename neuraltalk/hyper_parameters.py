# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/15 23:40


class HyperParameters():
    max_len = 34
    min_count = 1
    embedding_dim = 256
    batch_size = 32
    n_epochs = 20
    train_size = 1
    pkl_file = './feats_dict.pkl'
    dataset_path = './data/Flickr8k_Dataset/'
    token_file = './data/Flickr8k_text/Flickr8k.token.txt'
    train_images_file = './data/Flickr8k_text/Flickr_8k.trainImages.txt'
    test_images_file = './data/Flickr8k_text/Flickr_8k.testImages.txt'
    train_feats_file = './train_feats.npy'
    test_feats_file = './test_feats.npy'
