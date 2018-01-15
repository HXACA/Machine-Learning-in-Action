#-*- coding:utf-8 _*-
from numpy import *
""" 
@author:KING 
@file: bayes.py 
@time: 2018/01/13 
"""

def loadDataSet():
    postingList =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
        #取并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    #词集模型
    returnVec = [0]*len(vocabList)
    #创建全为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
            #标记出现的词
        else : print "%s未出现在我的词典中！" % word
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    #训练函数
    #文档矩阵，每篇类别标签的向量
    numTrainDocs = len(trainMatrix)
    #获取总文章数
    numWords = len(trainMatrix[0])
    #获取总词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #总比例
    p0Num = ones(numWords)
    #记录0类别的出现次数
    p1Num = ones(numWords)
    #记录1类别的出现次数,避免出现0
    p0Denom = 2.0
    p1Denom = 2.0
    #该类别总词数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    #避免出现下溢出，即出现太多比较小的数相乘
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    #概率相乘再相加
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

def bagOfWords2VecMN(vocabList, inputSet):
    #词袋模型
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def textParse(bigString):
    #文本切分
    import re
    listOfTokens = re.split(r'\W*',bigString)
    #正则表达式，只分割字母，数字
    return [tok.lower() for tok in listOfTokens if len(tok)>2]
    #转小写并只选择长度大于2的

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    #建立词列表
    trainingSet = range(50)
    print trainingSet
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #随机一个数字，并将对应文章放入测试集
        del(trainingSet[randIndex])
        #删除该数字
    #随机测试集
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pAb = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pAb) != classList[docIndex]:
            errorCount+=1
    print 'the error rate is : ',float(errorCount)/len(testSet)


