#coding=utf-8
"""使用CNN网络进行文本分类
    class CNNs:
        __init__(input_shape,nb_filters,nb_conv,nb_pool,batch_size,np_epoch,verbose)
            :param
                input_shape:输入维度，3D张量，实验中为文本词向量维度.默认(1,100,100)\n
                nb_filters:卷积核个数，实验中为两个卷积层，各3个卷积核.默认(3,3)\n
                nb_conv:卷积模板大小.默认(3,3)\n
                nb_pool:池化模板大小.默认(2,2)\n
                batch_size:训练批大小.默认300\n
                nb_epoch:训练迭代次数.默认100\n
                verbose:显示训练信息.默认1
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
        cnn=CNNs(verbose=2)\n
        cnn.packValidData()\n
        cnn.buildModel()\n
        cnn.train(Trained=False)
        
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

    def __init__(self,input_shape=(1,100,100),nb_filters=(3,3),nb_conv=(3,3),nb_pool=(2,2),batch_size=300, n_epoch=100, verbose=1):
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
        model.add(Conv2D(self.nb_filters[0],(self.nb_conv[0],self.nb_conv[1]),
                         input_shape=(1,self.input_shape[1], self.input_shape[2]),
                         data_format="channels_first"))
        model.add(Activation('relu'))
        # 下采样层S1
        model.add(MaxPooling2D(data_format="channels_first",pool_size=(self.nb_pool[0], self.nb_pool[1])))

        # 卷积层C2
        if self.nb_conv[1]==100:
            model.add(Conv2D(self.nb_filters[1], (self.nb_conv[0], 1), data_format="channels_first"))
        else:
            model.add(Conv2D(self.nb_filters[1],(self.nb_conv[0],self.nb_conv[1]),data_format="channels_first"))
        model.add(Activation('relu'))
        # 下采样层S2
        model.add(MaxPooling2D(data_format="channels_first",pool_size=(self.nb_pool[0], self.nb_pool[1])))

        model.add(Flatten())
        # 全连接层
        # model.add(Dense(100,kernel_initializer="normal"))
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
        valid = cPickle.load(open('../model/CNNs/valid.pkl', 'r'))
        vLabels = cPickle.load(open('../model/CNNs/vLabels.pkl', 'r'))
        t2=time.time()
        print 'load set finished.take time:%fs'% (t2-t1)
        model.fit_generator(self.gLoadTrainData(self.batch_size), steps_per_epoch=16,epochs=self.n_epoch,verbose=self.verbose,
                            validation_data=(valid, vLabels))
        t3=time.time()
        print 'train model finished.take time:%fs'% (t3-t1)
        model.save_weights('../model/CNNs/cnn_weight.h5')

    def gLoadTrainData(self, batch_size):
        """加载数据集，生成器

        """
        w2vmod = vecMod.Word2Vec.load('../model/W2V/wordVec.model')
        crtBatch = 0
        while 1:
            with open('../dataSet/rnn_train.txt', 'r') as f:
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
        w2vmod = vecMod.Word2Vec.load('../model/W2V/wordVec.model')
        validSet=np.zeros((600, 1, self.input_shape[1], self.input_shape[2]), dtype='float32')
        vLabels =np.zeros((600,6), dtype='float32')
        num=0
        with open('../dataSet/rnn_valid.txt', 'r') as f:
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

        cPickle.dump(validSet,open('../model/CNNs/valid.pkl','w'))
        cPickle.dump(vLabels,open('../model/CNNs/vLabels.pkl','w'))


if __name__ == '__main__':
    cnn=CNNs(verbose=2)
    # cnn.packValidData()
    cnn.buildModel()
    cnn.train(Trained=False)