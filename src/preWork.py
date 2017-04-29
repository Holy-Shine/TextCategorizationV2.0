# coding=utf-8
"""preWork.py

做一些准备工作

含有模块：
    divideDatabase      划分数据集、验证集、测试集
    crtCls2Terms        创建类别-词语字典
    crtCls2Text         创建类别-文章-文字字典
    crtPaper2words      创建word2vec的训练格式文件
    trainWordsVec()     训练词向量
生成文件
    dict.pkl                 形式{class1:[word1,word2...],class2:[...]}
    cls2Text.pkl         形式{class1:[text1,text2...],class2:[...]}
"""
import random
import sys,cPickle
import jieba
from toolFunc import segText,rmvStopWord
from gensim.models import word2vec as w2v
import multiprocessing

reload(sys)
sys.setdefaultencoding('utf-8')





#共8个类别
articleClass=['education',
              'entertainment',
              'it',
              'culture',
              'history',
              'military',
              'reading',
              'society&law']

def divideDatabase():
    """划分训练集、验证集、测试集.
    
    比例-8:1:1
    每个类别1000篇文章
    """
    dic={}
    for cls in articleClass:
        dic[cls]=[]
    file=open('database.txt','r')
    for line in file:
        item=line.split(" ")
        str=item[1].decode('utf-8')
        dic[item[0]].append(str)
    file.close()
    
    for cls in articleClass:
        print len(dic[cls])
    #以下几行，分别将训练、验证、测试集写入文件
    trainPath=open('train.txt','w')
    validPath=open("valid.txt",'w')
    testPath=open('test.txt','w')
    
    #shuffle
    shuffleList=[i for i in range(1000)]
    random.shuffle(shuffleList)
    print shuffleList
    
    for cls in articleClass:
        articleList=dic[cls]
        if len(articleList)<1000:
            continue
        for i in range(0,800):
            trainPath.write(cls+" "+articleList[shuffleList[i]])
        for i in range(800,900):
            validPath.write(cls+" "+articleList[shuffleList[i]])
        for i in range(900,1000):
            testPath.write(cls+" "+articleList[shuffleList[i]])          
#             
    testPath.close()
    validPath.close()
    trainPath.close()
      
def crtCls2Terms():
    """构建类别-词语字典
    
    """
    file=open('dataSet/train.txt','r')
    jieba.load_userdict('dict.txt')
    dic={}
    for line in file:
        tp=line.split(' ')
        if tp[0] not in dic:
            dic[tp[0]]=[]
        dic[tp[0]].extend(segText(tp[1]))
               
    for key in dic.keys():
        dic[key]=list(set(dic[key]))
    
    #载入停用词
    stopWords=[word.replace("\r\n","").replace(" ","")  for  word in open('stopwords.txt','r').readlines()]
    for key in dic.keys():
        print key,len(dic[key])
        dic[key]=rmvStopWord(dic[key], stopWords)
        print key,len(dic[key])
    cPickle.dump(dic, open("model/dict.pkl",'w'), protocol=0)
    
def crtCls2Text():
    """创建类别-Text字典
    
    """
    file=open('dataSet/train.txt','r')
    cat={}
    for line in file:
        tp=line.split(' ')
        if tp[0] not in cat:
            cat[tp[0]]=[]
        cat[tp[0]].append(tp[1])
    
    file.close()
    cPickle.dump(cat, open('model/cls2Text.pkl','w'), protocol=0)

         
    
def train(trainPath):
    pass


#以下是word2vec的相关处理内容
def crtPaper2words():
    '''按照word2vec的需求，处理成一篇文章以个word list的形式
    
    输出文件：
        w2v_paper2words.txt
    '''
    sf=open('../preTrain/w2v_paper2words.txt','w')
    jieba.load_userdict('dict.txt')
    with open('dataSet/train.txt','r') as f:
        i=0
        for line in f:
            arr=line.split(' ')
            wordList=[]
            if len(arr)>1:
                wordList=segText(arr[1])
            for word in wordList:
                sf.write(word+' ')
            sf.write('\n')

def trainWordsVec(dimension=500,window=5,min_count=3):
    '''使用word2vec训练语料
    
    文件输入:
        preTrain/w2v_paper2words.txt
    模型输出：
        model/wordVec.model
    '''
    #输入语料
    inq='../preTrain/w2v_paper2words.txt'
    #输出模型1
    out1='model/wordVec.model'
    # out2为原始c版本word2vec的vector格式的模型
    out2 = 'model/wordVec.vector'
    model=w2v.Word2Vec(w2v.LineSentence(inq),
                       size=dimension, window=window, min_count=min_count,
                       workers=multiprocessing.cpu_count())
    model.save(out1)

def rnn_divideDatabase():
    """rnn的数据集切分策略
    输出：
        rnn_train.txt
        rnn_test.txt
        rnn_valid.text
    """
    jieba.load_userdict('dict.txt')
    def openAndSave(inf,outf):
        outf=open(outf,'w')
        with open(inf, 'r') as f:
            for line in f:
                arr = line.split(' ')
                wordList = []
                if len(arr) > 1:
                    wordList = segText(arr[1])
                    outf.write(arr[0]+'::')
                for word in wordList:
                    outf.write(word + ' ')
                outf.write('\n')
        outf.close()
    openAndSave('dataSet/train.txt','dataSet/rnn_train.txt')
    openAndSave('dataSet/test.txt','dataSet/rnn_test.txt')
    openAndSave('dataSet/valid.txt','dataSet/rnn_valid.txt')

if __name__ =='__main__':
 
    
    #1、划分训练集合、验证集、测试集
#    file=open('database.txt','r')
#    # divideDatabase()
#    file.close()
    #2、对训练集分词、去停用词，构建不同类别的词语集合字典dic
#     crtCls2Terms()
#         crtCls2Text()
#     crtPaper2words()
    trainWordsVec(dimension=100)

    # rnn_divideDatabase()


    pass