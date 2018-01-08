# -- coding: utf-8 --
from math import log
import operator

#计算给定数据集的香农熵，熵的变化就是无序度的变化
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts ={}
    #记录所有的label
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] +=1
        #对label进行计数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        #此label出现的次数/总数据条数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    #新的所求list
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #保存前axis列
            reducedFeatVec.extend(featVec[axis+1:])
            #保存axis之后的列
            retDataSet.append(reducedFeatVec)
            #加入结果list
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    #获得特征数量
    baseEntropy = calcShannonEnt(dataSet)
    #计算无序度量值
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #遍历所有特征
        featList = [example[i] for example in dataSet]
        #featList为当前列的所有特征
        uniquaVals = set(featList)
        newEntropy = 0.0
        for value in uniquaVals:
            #以当前特征值中不同的value分割数据集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntropy
        #相当于对分割出的数据集求熵并相加，取最小
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    #出现次数最多的分类名称
    classCount ={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),\
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    #建树，用字典变量存储树结构
    classList = [example[-1] for example in dataSet]
    #当前dataSet中的所有分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #只有一个分类
    if len(dataSet[0])==1 :
        return majorityCnt(classList)
    #只剩一个特征返回出现次数最多的label
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #获取最佳特征的名称
    myTree = {bestFeatLabel: {}}
    tlabels = labels[:]
    del(tlabels[bestFeat])
    #删去这一个特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues =set(featValues)
    #获取该特征的所有value
    for value in uniqueValues:
        subLabels = tlabels[:]
        #复制了类标签，这样不会改变原类列表的内容
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #按照最佳分割数据集的方法分割数据集并得到下一层的决策树
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    #转为index
    for key in secondDict.keys():
        if(testVec[featIndex]==key):
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    #序列化对象,即保存计算出的决策树
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def solve(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    storeTree(lensesTree,'lensesTree.txt')
    return lenses

