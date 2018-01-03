# -- coding: utf-8 --
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
#定义叶节点，判断框和箭头的样式

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,\
                            xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',\
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
    #注解函数，可以用来对坐标中的数据进行注解，让人更清晰的得知坐标点得意义，现在对其参数作用进行分析：
    #xy -- 为点的坐标
    #xytext -- 为注解内容位置坐标，当该值为None时，注解内容放置在xy处
    #xycoords and textcoords 是坐标xy与xytext的说明，若textcoords=None，则默认textNone与xycoords相同，若都未设置，默认为data，
    #arrowprops -- 用于设置箭头的形状，类型为字典类型

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    #清空绘图区
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=True, **axprops)
    #frameon为外边框
    #plotNode( U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    #plotNode( U'叶节点', (0.8, 0.1), (0.3, 0.8) , leafNode)
    # #Unicode解码问题
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    #记录当前x,y坐标
    plotTree(inTree, (0.5, 1.0), ' ')
    plt.show()

#获取叶节点的数目
def getNumLeafs(myTree) :
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #判断子节点是否为字典类型
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else: thisDepth =1
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    #填充文本信息
    xMid = (parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    #求父节点与子节点间的中点
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    #标注点坐标
    plotMidText(cntrPt, parentPt, nodeTxt)
    #补充中间内容
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    #画出当前决策节点
    secondDict = myTree[firstStr]
    #取子节点
    plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD
    #计算出下一层y的坐标
    #偏移y坐标
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
            #非叶节点递归绘制，以中心点为父节点坐标位置
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,plotTree.yOff), cntrPt, leafNode)
            #绘制叶节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt,str(key))
            #补充中间内容
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    #返回上一层时恢复y