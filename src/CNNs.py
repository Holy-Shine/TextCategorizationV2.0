#coding=utf-8
"""使用CNN网络进行文本分类
"""
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
import gensim.models as vecMod
import numpy as np
import time
import cPickle
class CNNs(object):

    def __init__(self,input_shape=(1,100,100),nb_filters=(3,3),nb_conv=3,nb_pool=2,batch_size=300, n_epoch=100, verbose=1):
        self.nb_filters=nb_filters
        self.nb_conv=nb_conv
        self.input_shape=input_shape
        self.nb_pool=nb_pool
        self.n_epoch = n_epoch
        self.verbose = verbose
        self.batch_size = batch_size


    def buildModel(self):
        model = Sequential()
        # 卷积层C1
        model.add(Conv2D(self.nb_filters[0],(self.nb_conv,self.nb_conv),
                         input_shape=(1,self.input_shape[1], self.input_shape[2]),
                         data_format="channels_first"))
        model.add(Activation('relu'))
        # 下采样层S1
        model.add(MaxPooling2D(data_format="channels_first",pool_size=(self.nb_pool, self.nb_pool)))

        # 卷积层C2
        model.add(Conv2D(self.nb_filters[1],(self.nb_conv,self.nb_conv),data_format="channels_first"))
        model.add(Activation('relu'))
        # 下采样层S2
        model.add(MaxPooling2D(data_format="channels_first",pool_size=(self.nb_pool, self.nb_pool)))

        model.add(Flatten())
        # 全连接层
        model.add(Dense(100,kernel_initializer="normal"))
        model.add(Activation('relu'))
        model.add(Dense(6))
        model.add(Activation('softmax'))

        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd)

        self.model=model

    def train(self, Trained=False):
        t1=time.time()
        print 'training model...'
        model = self.model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        t1=time.time()
        print 'load valid set...'
        valid = cPickle.load(open('model/rnnModel/valid.pkl', 'r'))
        vLabels = cPickle.load(open('model/rnnModel/vLabels.pkl', 'r'))
        t2=time.time()
        print 'load set finished.take time:%fs'% (t2-t1)
        model.fit_generator(self.gLoadTrainData(self.batch_size), steps_per_epoch=16,epochs=self.n_epoch,verbose=self.verbose,
                            validation_data=(valid, vLabels))
        t3=time.time()
        print 'train model finished.take time:%fs'% (t3-t1)
        model.save_weights('lstm_weight.h5')

    def gLoadTrainData(self, batch_size):
        """加载数据集，生成器

        """
        w2vmod = vecMod.Word2Vec.load('model/wordVec.model')
        crtBatch = 0
        while 1:
            with open('dataSet/rnn_train.txt', 'r') as f:
                num = 0
                trainBatch = np.zeros((batch_size, 1, self.input_shape[1], self.input_shape[2]), dtype='float32')
                trainLabel = np.zeros((batch_size, 6), dtype='float32')
                for line in f:
                    label = [0.0] * 6
                    sample = [[0.0] * self.input_shape[1]] * self.input_shape[2]
                    arr = line.split('::')
                    if len(arr) > 1:
                        label[int(arr[0]) - 2] = 1.0
                        wordsL = arr[1].split(' ')
                        i = 0
                        for index in xrange(min(self.input_shape[1], len(wordsL))):
                            try:
                                sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()
                            except:
                                pass
                                # if wordsL[index].decode('utf-8') in w2vmod:
                                #     sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()
                        trainBatch[num, :, :, :] = np.array(sample)
                        trainLabel[num, :] = np.array(label)
                        num+=1
                        if num % batch_size == 0:
                            crtBatch += 1
                            yield (trainBatch, trainLabel)
                            trainBatch = np.zeros((batch_size, 1, self.input_shape[1], self.input_shape[2]),
                                                  dtype='float32')
                            trainLabel = np.zeros((batch_size, 6), dtype='float32')
                            num = 0

    def packValidData(self):
        """组织验证集数据,生成网络输入所需数据格式
        """
        w2vmod = vecMod.Word2Vec.load('model/wordVec.model')
        validSet=np.zeros((600, 1, self.input_shape[1], self.input_shape[2]), dtype='float32')
        vLabels =np.zeros((600,6), dtype='float32')
        num=0
        with open('dataSet/rnn_valid.txt', 'r') as f:
            for line in f:
                label = [0.0] * 6
                sample = [[0.0] * self.input_shape[1]] * self.input_shape[2]
                arr = line.split('::')
                if len(arr) > 1:
                    label[int(arr[0]) - 2] = 1.0
                    wordsL = arr[1].split(' ')
                    for index in xrange(min(self.input_shape[1], len(wordsL))):
                        try:
                            sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()
                        except:
                            pass
                validSet[num,:,:,:]=np.array(sample)
                vLabels[num,:]=np.array(label)
                num+=1

        cPickle.dump(validSet,open('model/rnnModel/valid.pkl','w'))
        cPickle.dump(vLabels,open('model/rnnModel/vLabels.pkl','w'))

if __name__ == '__main__':
    cnn=CNNs(verbose=2)
    # cnn.packValidData()
    cnn.buildModel()
    cnn.train(Trained=False)