# coding=utf-8
"""使用RNNs进行文本分类

输入：
    word2vec向量
    train.txt\vaild.txt\test.txt
输出：
    预测模型
"""
from keras.models import Sequential
from keras.layers import Masking, Dropout, Dense
from keras.layers import LSTM
import gensim.models as vecMod
import numpy as np
import cPickle
import time

class RNNs(object):
    def __init__(self,inputShape=400,maxLenth=500,batch_size=256,n_epoch=20,verbose=1,shuffle=True):
        self.inputShape = inputShape
        self.maxlenth = maxLenth
        self.n_epoch=20
        self.verbose=1
        self.shuffle=True
        self.batch_size=256

    def buildModel(self, lstm=False):
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(self.maxLenth, self.inputShape)))
        model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2, input_shape=(self.maxLenth, self.inputShape)))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        self.model = model

    def train(self, train, label, valid, Trained=False):
        model = self.model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        valid=cPickle.load(open('valid.pkl','r'))
        vLabels=cPickle.load(open('vLabels.pkl','r'))
        model.fit_generator(self.gLoadTrainData(),self.batch_size,self.batch_size,self.verbose,validation_data=(valid,vLabels))

    def gLoadTrainData(self):
        """加载数据集，生成器
            
        """
        w2vmod = vecMod.Word2Vec.load('model/wordVec.model')
        with open('dataSet/rnn_train.txt', 'r') as f:
            for line in f:
                label = [0.0] * 6
                sample = [[0.0] * self.inputShape] * self.maxlenth
                arr = line.split('::')
                if len(arr) > 1:
                    label[int(arr[0]) - 2] = 1.0
                    wordsL = arr[1].split(' ')
                    i = 0
                    for index in xrange(min(self.maxlenth,len(wordsL))):
                        if wordsL[index].decode('utf-8') in w2vmod:
                            sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()

                yield (np.array(sample,dtype='float32'),np.array(label,dtype='float32'))


    def packData(self):
        # """组织验证集数据,生成网络输入所需数据格式
        # """
        # print('begin packing trainData...')
        # t1=time.time()
        # self.trainL, self.tLabels = self.getDataFromDb('dataSet/rnn_train.txt')
        # t2=time.time()
        # print('pack trainData finished,take time:%d'%t2-t1)
        #
        # print('begin packing validData...')
        # self.validL, self.vLabels = self.getDataFromDb('dataSet/rnn_valid.txt')
        # t3=time.time()
        # print('pack validData finished,take time:%d'%t3-t2)
        #
        # cPickle.dump(self.trainL, open('model/rnnModel/train.pkl'))
        # cPickle.dump(self.tLabels, open('model/rnnModel/tLabels.pkl'))
        # cPickle.dump(self.validL, open('model/rnnModel/valid.pkl'))
        # cPickle.dump(self.vLabels, open('model/rnnModel/vLabels.pkl'))
        pass

if __name__ == '__main__':
    module = RNNs()
    module.packData()


