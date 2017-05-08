# -*- coding: utf-8 -*-

import numpy as np


class Poetry:
    def __init__(self, title = '', author = '', 
                content = ''):
        self.title = title
        self.author = author
        self.content = content

    def __repr__(self):
        return "title: " + str(self.title)  + ", content: " + str(self.content) + '\n'




class Poetries:
    '''
    古诗的数据集。将每一首古诗切分成5个字5个字的小段，即每次训练都是5个字，
    这里要生成直接可用的数据集
    '''
    def __init__(self, pts):
        '''
        pts: list, 是一个由编码过的Poetry构成的列表
        return:
        data_set: ndarray, 用于训练的数据
        target: ndarray, shape和data_set是一样的，目标输出
        '''
        self.sample_len = 8
        self.data_set = None
        self.target = None
        for i in xrange(len(pts)):
            d, t = self._cut(pts[i])
            if self.data_set is None:
                self.data_set = d
                self.target = t
            else :
                self.data_set = np.concatenate([self.data_set, d], axis=0)
                self.target = np.concatenate([self.target, t], axis=0)


    def _cut(self, poetry, dtype = np.int32):
        '''
        poetry: class Poetry,古诗必须是经过编码的，即用数字表示的
        将古诗的题目和内容并在一起，然后进行切分。每一小段的长度由
        self.sample_len来控制
        '''
        sample_len = self.sample_len
        length = len(poetry.title) + len(poetry.content)
        print 'poetry length: ', length
        ret_data = np.zeros((length - sample_len , sample_len), dtype=dtype)
        ret_target = np.zeros((length - sample_len , sample_len), dtype=dtype)
        data = poetry.title + poetry.content
        for i in xrange(length - sample_len ):
            ret_data[i, :] = data[i:i + sample_len]
            ret_target[i,:] = data[i+1:i + sample_len + 1]
        return ret_data, ret_target


    def __repr__(self):
        return str(self.data_set.shape)

    
    def __getitem__(self, index):
        return self.data_set[index], self.target[index]