# coding=utf-8
"""使用RNNs进行文本分类

    class RNNs:
        __init__(inputShape,maxLenth,batch_size,n_epoch,verbose, shuffle)
            :param
                inputShape:输入维度，实验中为词向量维度.默认300\n
                maxLenth:时间步最长个数，实验中为最大文本词语个数
                batch_size:训练批大小.默认300\n
                nb_epoch:训练迭代次数.默认100\n
                verbose:显示训练信息.默认1\n
                shuffle:是否打乱数据.默认1
        bulidModel():
            构建模型
        train():
            训练模型
        gLoadTrainData(batchsize):
            训练集加载生成器
        packValidData():
            组装验证集合
            
    文件依赖（需要下列模块产生的模型文件）：
        preWork.py:
            w2v_crtPaper2words  产生dataSet/w2v_paper2words.txt
            w2v_trainWordsVec() 产生model/W2V/wordVec.model
    e.g.
        lstm=RNNs(inputShape=100,maxLenth=100)\n
        lstm.packValidData()\n
        lstm.buildModel()\n
        lstm.train(Trained=False)
"""
from keras.models import Sequential
from keras.layers import Masking, Dropout, Dense,Embedding
from keras.layers import LSTM
import gensim.models as vecMod
import numpy as np
import cPickle
import time


class RNNs(object):
    def __init__(self, inputShape=400, maxLenth=400, batch_size=300, n_epoch=20, verbose=1, shuffle=True):
        self.inputShape = inputShape
        self.maxLenth = maxLenth
        self.n_epoch = n_epoch
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size

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
        valid = cPickle.load(open('../model/RNNs/valid.pkl', 'r'))
        vLabels = cPickle.load(open('../model/RNNs/vLabels.pkl', 'r'))
        t2=time.time()
        print 'load set finished.take time:%fs'% (t2-t1)
        model.fit_generator(self.gLoadTrainData(self.batch_size), steps_per_epoch=16,epochs=self.n_epoch,verbose=self.verbose,
                            validation_data=(valid, vLabels))
        t3=time.time()
        print 'train model finished.take time:%fs'% (t3-t1)
        model.save_weights('../model/RNNs/lstm_weight.h5')

    def gLoadTrainData(self,batch_size):
        """加载数据集，生成器

        """
        w2vmod = vecMod.Word2Vec.load('../model/W2V/wordVec.model')
        while 1:
            with open('../dataSet/rnn_train.txt', 'r') as f:
                num=0
                batchList,batchLabels=[],[]
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
                        num+=1
                        batchList.append(sample)
                        batchLabels.append(label)
                        TrainBatch=np.array(batchList,dtype='float32')
                        TrainLabel=np.array(batchLabels,dtype='float32')
                        if num%batch_size==0:
                            print 'add a sample'
                            yield (TrainBatch,TrainLabel)
                            batchList, batchLabels = [], []



    def packValidData(self):
        """组织验证集数据,生成网络输入所需数据格式
        """
        w2vmod = vecMod.Word2Vec.load('../model/W2V/wordVec.model')
        validSet,vLabels=[],[]
        with open('../dataSet/rnn_valid.txt', 'r') as f:
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

        cPickle.dump(np.array(validSet),open('../model/RNNs/valid.pkl','w'))
        cPickle.dump(np.array(vLabels),open('../model/RNNs/vLabels.pkl','w'))


if __name__ == '__main__':
    lstm=RNNs(inputShape=100,maxLenth=100)
    lstm.packValidData()
    lstm.buildModel()
    lstm.train(Trained=False)


