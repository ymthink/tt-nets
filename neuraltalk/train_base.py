# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/4/27 12:22


from NeuralTalk import NeuralTalk


if __name__ == '__main__':
    nt = NeuralTalk(is_TT=False)
    nt.train(validation=0)
    # nt.eval()