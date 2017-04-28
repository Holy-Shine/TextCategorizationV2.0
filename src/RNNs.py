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
    def __init__(self, inputShape=400, maxLenth=400, batch_size=256, n_epoch=20, verbose=1, shuffle=True):
        self.inputShape = inputShape
        self.maxLenth = maxLenth
        self.n_epoch = 20
        self.verbose = 1
        self.shuffle = True
        self.batch_size = 256

    def buildModel(self, lstm=False):
        t1=time.time()
        print 'building model...'
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(self.maxLenth, self.inputShape)))
        model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2, input_shape=(self.maxLenth, self.inputShape)))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        self.model = model
        t2=time.time()
        print 'bulid model finished.take time:%fs'% (t2-t1)


    def train(self,Trained=False):
        t1=time.time()
        print 'training model...'
        model = self.model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        t1=time.time()
        print 'load valid set...'
        valid = cPickle.load(open('model/rnnModel/valid.pkl', 'r'))
        vLabels = cPickle.load(open('model/rnnModel/vLabels.pkl', 'r'))
        t2=time.time()
        print 'load set finished.take time:%fs'% (t2-t1)
        model.fit_generator(self.gLoadTrainData(), steps_per_epoch=4800,epochs=self.n_epoch,verbose=self.verbose,
                            validation_data=(valid, vLabels))
        t3=time.time()
        print 'train model finished.take time:%fs'% (t3-t1)
        model.save_weights('lstm_weight.h5')

    def gLoadTrainData(self):
        """加载数据集，生成器

        """
        w2vmod = vecMod.Word2Vec.load('model/wordVec.model')
        while 1:
            with open('dataSet/rnn_train.txt', 'r') as f:
                for line in f:
                    label = [0.0] * 6
                    sample = [[0.0] * self.inputShape] * self.maxLenth
                    arr = line.split('::')
                    if len(arr) > 1:
                        label[int(arr[0]) - 2] = 1.0
                        wordsL = arr[1].split(' ')
                        i = 0
                        for index in xrange(min(self.maxLenth, len(wordsL))):
                            try:
                                sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()
                            except:
                                pass
                            # if wordsL[index].decode('utf-8') in w2vmod:
                            #     sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()
                        print 'add a sample'
                        yield (np.array(sample, dtype='float32'), np.array(label, dtype='float32'))

    def packValidData(self):
        """组织验证集数据,生成网络输入所需数据格式
        """
        w2vmod = vecMod.Word2Vec.load('model/wordVec.model')
        validSet,vLabels=[],[]
        with open('dataSet/rnn_valid.txt', 'r') as f:
            for line in f:
                label = [0.0] * 6
                sample = [[0.0] * self.inputShape] * self.maxLenth
                arr = line.split('::')
                if len(arr) > 1:
                    label[int(arr[0]) - 2] = 1.0
                    wordsL = arr[1].split(' ')
                    for index in xrange(min(self.maxLenth, len(wordsL))):
                        try:
                            sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()
                        except:
                            pass
                    validSet.append(sample)
                    vLabels.append(label)

        cPickle.dump(validSet,open('model/rnnModel/valid.pkl','w'))
        cPickle.dump(vLabels,open('model/rnnModel/vLabels.pkl','w'))


if __name__ == '__main__':
    lstm=RNNs()
    # lstm.packValidData()
    lstm.buildModel()
    lstm.train(Trained=False)


