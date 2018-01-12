#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: kMeans.py 
@time: 2018/01/08 
"""
from numpy import *
import urllib
import json
from time import sleep

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
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
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
            #取平方更为重视距离较远的点
        #print centroids
        for cent in range(k):
            #遍历所有的质心
            ptsInclust = dataset[nonzero(clusterAssment[:, 0].A == cent)[0]]
            #获得第cent簇中的所有点
            centroids[cent, :] = mean(ptsInclust, axis=0)
            #按照列方向进行均值计算，求均值
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    #分配结果以及误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #按照列方向求均值并转为列表
    centList = [centroid0]
    #将所有的点作为一簇，储存所有的质心
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    #计算每个点与质心的误差
    while(len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            #遍历所有的质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            #选出对应簇的点
            centroidMat, splitClusterAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #将给定簇分为2聚
            sseSplit = sum(splitClusterAss[:, 1])
            #计算总误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            #计算未分割部分的总误差
            print "sseSplit:, and notSplit: ", sseSplit, sseNotSplit
            if (sseSplit+sseNotSplit) < lowestSSE:
                bestCentTosplit = i
                bestNewsCent = centroidMat
                bestClustAss = splitClusterAss.copy()
                lowestSSE = sseNotSplit+sseSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentTosplit
        #根据索引不同修改簇编号
        print 'the bestCentToSplit is: ', bestCentTosplit
        print 'the len of bestClustAss is ', len(bestClustAss)
        centList[bestCentTosplit] = bestNewsCent[0, :].tolist()[0]
        centList.append(bestNewsCent[1, :].tolist()[0])
        #用分割出的新的两簇替换原来的一簇
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentTosplit)[0], :] = bestClustAss
        #记录新的结果和误差
    #print centList
    return mat(centList), clusterAssment

def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'dj0yJmk9TEZraUVKYVNFeDFrJmQ9WVdrOVFtZHVOVEF5Tm04bWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD0wOQ--'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem+url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())

def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readLines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (line, lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error"
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):
    a = sin(vecA[0, 1]*pi/180)*sin(vecB[0, 1]*pi/180)
    b = cos(vecA[0, 1]*pi/180)*cos(vecB[0, 1]*pi/180)*cos(pi*(vecB[0, 0]-vecA[0, 0])/180)
    return arccos(a+b)*6371.0
    #球面余弦定理

import matplotlib
import matplotlib.pyplot as plt

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    #调用二分kMeans
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    #坐标 ，x方向的长度，y方向的长度
    scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
    #绘制的点的样式
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0',**axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon = False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0],:]
        markerStyle = scatterMarkers[i%len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                    ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],\
                myCentroids[:,1].flatten().A[0],marker='+',s=300)
    plt.show()