# -- coding: utf-8 --
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

def classify0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #返回数组第0维的长度
    diffMat = tile(inx, (dataSetSize,1))-dataSet
    #重复inx至dataSet的大小
    sqDiffMat = diffMat**2
    #平方计算
    sqDistances = sqDiffMat.sum(axis=1)
    #行求和
    distances = sqDistances**0.5
    #开方运算
    sortedDistIndicies = distances.argsort()
    #取下标的排序
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1;
        #在字典中查找voteIlabel的值
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #按照第一个域的数据逆序排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    #按行读入
    numberOfLines = len(arrayOLines)
    #获得行数
    returnMat = zeros((numberOfLines,3))
    #构造同大小的0矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        #去空格
        listFromLine = line.split('\t')
        #使用'\t'分割
        returnMat[index, :] = listFromLine[0:3]
        #储存每行的数据
        classLabelVector.append(int(listFromLine[-1]))
        #取列表倒数第一个
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet) :
    minVals = dataSet.min(0)
    #取每列最小值
    maxVals = dataSet.max(0)
    #取每列最大值
    ranges = maxVals-minVals
    #得出每列的范围
    normDataSet = zeros(shape(dataSet))
    #构造结果数组
    m = dataSet.shape[0]
    #得到dataSet的行数
    normDataSet = dataSet-tile(minVals, (m, 1))
    #newValue = (oldValue-min)/(max-min)
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest() :
    hoRatio = 0.10
    datingDataMat ,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    #测试个数
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m],3)
        #取第i行数据，以从numTestVecs到m行为训练集，k为3，进行k-近邻算法分类
        print "the classifier came back with : %d , the real answer is : %d"\
                 % (classifierResult, datingLabels[i])
        if(classifierResult!=datingLabels[i]) : errorCount+=1
        #记录错误次数
    print "the total error rate is : %f" % (errorCount/float(numTestVecs))
    #输出错误比例

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses' ]
    percentTats = float(raw_input(\
        "percentage of time spent playing video games?"))
    #从键盘输入
    ffmiles = float(raw_input(\
        "frequent fliter miles earned per year?"))
    iceCream = float(raw_input(\
        "Liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #从文件获取数据
    normMat , ranges, minVals = autoNorm(datingDataMat)
    #归一化数据
    inArr = array([ffmiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    #k-近邻算法处理
    print "You will probably like this person : ", resultList[classifierResult - 1]
    #输出结果

