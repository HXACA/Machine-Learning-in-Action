# -- coding: utf-8 --
from math import log
import operator

#计算给定数据集的香农熵，熵的变化就是无序度的变化
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts ={}
   # print "%d" %numEntries
    for featVec in dataSet:
        currentLabel = featVec[-1]
       # print "%s" %currentLabel
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
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
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0])==1 :
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues =set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
