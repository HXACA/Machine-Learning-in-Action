#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: svmMLiA.py 
@time: 2018/02/11 
"""
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

class opStruct:
    #一个适合的数据结构 而非面向对象
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

def openCVtest():
    img = cv2.imread("./9.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print shape(thresh1)
    m,n=shape(thresh1)
    for i in range(m):
        for j in range(n):
            if thresh1[i][j]!=0:
                thresh1[i][j]=int(1)
            else:
                thresh1[i][j] =int(0)
    savetxt('9.txt',array(thresh1),fmt='%d')

def calcEK(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek = fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek = calcEK(oS,k)
            deltaE = abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK = k;maxDeltaE = deltaE;Ej = Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej = calcEK(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek = calcEK(oS,k)
    oS.eCache[k] = [1,Ek]

def kernelTrans(X,A,kTup):
    #kTup为描述核函数的类型
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:]-A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
        #numpy的除法是直接展开运算而非求逆
    else:raise NameError('Houston We Have a problem -- That Kernel is not recognize')
    return K

def testRbf(k1 = 1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]#支持向量矩阵
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n=shape(dataMat)
    errorCount =0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount+=1
    print "the training error rate is: %f%%" %(float(errorCount)/m*100)
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr);labelMat=mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f%%" % (float(errorCount) / m*100)
    show(dataArr,labelArr,alphas)

def img2vector(filename):
    #把32*32的二进制图像转为1*1024的向量
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        #逐行读取
        lineStr=lineStr.replace(' ','')
        #print lineStr
        for j in range(32):
            #print i, j, len(lineStr), int(lineStr[j])
            returnVect[0, 32*i+j]=int(lineStr[j])
            #强转int否则为字符
    return returnVect

def loadImage(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNmaeStr = trainingFileList[i]
        fileStr = fileNmaeStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr==9:hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' %(dirName,fileNmaeStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImage('trainingDigits')#训练集
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d support vectors" %(shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
    print U"训练时的错误率是 %f%%" %(float(errorCount)/m*100)
    dataArr, labelArr = loadImage('testDigits')  # 测试集
    dataMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print U"测试时的错误率是 %f%%" % (float(errorCount) / m * 100)

def innerL(i,oS):
    #寻找决策边界
    Ei = calcEK(oS,i)
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or\
        ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L = max(0,oS.alphas[j]-oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print "L==H";return 0
        #eta = 2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        eta = 2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0 : print "eta>=0";return 0
        oS.alphas[j] -=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):print "j not moving enough";return 0
        oS.alphas[i] +=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        #b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        #b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0<oS.alphas[i]) and (oS.C > oS.alphas[i]):oS.b = b1
        elif(0<oS.alphas[j]) and (oS.C>oS.alphas[j]):oS.b = b2
        else:oS.b = (b1+b2)/2.0
        return 1
    else:return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ('lin',0)):
    oS = opStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True;alphaPairsChanged = 0
    while (iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print "fullSet,iter:%d i :%d , pairs changed %d" %(iter,i,alphaPairsChanged)
            iter+=1
        else:
            nonBoundIs = nonzero((oS.alphas.A>0) * (oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged +=innerL(i,oS)
                print "non-bound,iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged)
            iter+=1
        if entireSet:entireSet=False
        elif (alphaPairsChanged==0):entireSet=True
        print "iteration number: %d" %iter
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr);labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j = int(random.uniform(0,m))
    return j


def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def show(dataMatIn,classLabels,alphas):
    xcord1 = [];ycord1=[]
    xcord2 = [];ycord2=[]
    xcord3 = [];ycord3 = []
    m = shape(classLabels)[0]
    dataMatIn = array(dataMatIn)
    for i in range(m):
        if(alphas[i]>0.0):
            xcord3.append(dataMatIn[i, 0]);ycord3.append(dataMatIn[i, 1])
        elif(classLabels[i]==-1):
            xcord1.append(dataMatIn[i,0]);ycord1.append(dataMatIn[i,1])
        else:
            xcord2.append(dataMatIn[i,0]);ycord2.append(dataMatIn[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    ax.scatter(xcord3, ycord3, s=30, c='black')
    plt.show()


def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    #数据集，类别标签，常数C，容错率，迭代次数
    dataMatrix = mat(dataMatIn);labelMat = mat(classLabels).transpose()
    #转为矩阵
    b=0;m,n = shape(dataMatrix)
    #获得矩阵大小
    alphas = mat(zeros((m,1)))
    #初始化alpha向量
    iter = 0
    #计数器
    while(iter<maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #对数据集中每一个向量
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            #multiply 矩阵乘法 .T 转置
            Ei = fXi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i]<C)) or ((labelMat[i]*Ei > toler) and (alphas[i]>0)):
                #该数据向量可以被优化
                j = selectJrand(i,m)
                #随机选择另外一个数据向量
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i]!=labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print "L==H";continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0 :
                    print "eta>=0";continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001): print "J not moving enough";continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                #对i进行修改 修改量相同 方向相反
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                #常数项
                if (0<alphas[i]) and (C>alphas[i]) : b=b1
                elif (0<alphas[j]) and(C>alphas[j]) : b=b2
                else:b = (b1+b2)/2.0
                alphaPairsChanged +=1
                print "iter: %d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged==0):iter+=1
        else: iter=0
        print "iteration number : %d" % iter
    return b,alphas


