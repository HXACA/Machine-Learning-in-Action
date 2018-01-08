#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: kMeans.py 
@time: 2018/01/08 
"""
from numpy import *

def loadDataSet(filename):
    dataMat=[]
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #以tab进行分割
        fltLine = map(float, curLine)
        #对每个元素进行float强转
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))
    #计算两个向量的欧氏距离

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    #数据的列数
    centroids = mat(zeros((k, n)))
    #构造k行n列的全零矩阵
    for j in range(n):
        minJ = min(dataSet[:, j])
        #当前列最小值
        #print "%d" %minJ
        rangeJ = float(max(dataSet[:, j])-minJ)
        #当前列数据范围
        centroids[:, j]=minJ+ rangeJ*random.rand(k, 1)
        #表示随机出一个k*1的array，这里用rangeJ与随机数相乘，保证结果在范围之内
    return centroids

def kMeans(dataset, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataset)[0]
    clusterAssment = mat(zeros((m, 2)))
    #簇分配结果，第一列为索引，第二列为误差
    centroids = createCent(dataset, k)
    #随机创建k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                #计算第i个点与k个质心的分别的距离，求最小的距离
                distJI = distMeas(centroids[j, :], dataset[i, :])
                if distJI<minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                #最小值的索引发生了变化
                 clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        #print centroids
        for cent in range(k):
            #遍历所有的质心
            ptsInclust = dataset[nonzero(clusterAssment[:, 0].A == cent)[0]]
            #获得第cent簇中的所有点
            print ptsInclust
            centroids[cent, :] = mean(ptsInclust, axis=0)
            #按照列方向进行均值计算，求均值
    return centroids, clusterAssment