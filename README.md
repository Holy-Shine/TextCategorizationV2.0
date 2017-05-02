# 工程说明
## 包含模块
### 1-朴素贝叶斯
>NavieBayes.py
	
	#使用说明
	"""
	使用朴素贝叶斯进行模型预测
	P(C|x1,x2,x3....xn)正比于P(C)*P(x1|C)*P(x2|C)*P(x3|C)*...*P(xn|C)
	文件依赖：
    	preWork.py
        	nb_crtCls2Text 产生/model/NB/cls2Text.pkl
        	dataSet/test.txt
	"""
	#e.g.
	cls2Text = cPickle.load(open("../model/NB/cls2Text.pkl", 'r'))  # 形式{class1:[Text1,Text2...],class2:[...]}
    test = open('../dataSet/test.txt', 'r')
    errorNum = 0
    allNum = 0
    for line in test:
        allNum += 1
        truelabel = line.split(' ')[0]
        text = line.split(' ')[1]
        predictLable = predict(text, cls2Text)
        print truelabel, predictLable
        if (predictLable != truelabel):
            errorNum += 1

    print  "acc=%.2f%%" % (float(allNum - errorNum) * 100 / allNum)
        
### 2-RNN(LSTM)
>RNNs.py

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
### 3-CNN
>CNNs.py

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

