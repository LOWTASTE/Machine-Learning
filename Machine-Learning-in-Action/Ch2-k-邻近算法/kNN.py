from numpy import *
# 运算符模块
import operator
from os import listdir

'''
    example - 1
'''


def createDataSet():
    """

    :return: 坐标和标签
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# group, labels = createDataSet()


def classify0(inX, dataSet, labels, k):
    """

    :param inX: 待分类的数据
    :param dataSet: 用于分类的数据集
    :param labels: 数据集的标签
    :param k: k - 近邻
    :return: 分类的结果标签
    """
    # shape[0] 理解为第一维，shape[n] 理解为第 n + 1 维
    # ([[[1,2,3] , [4,5,6]]])，这是个三维数组，shape[0] = 1, shape[1] = 2, shape[2] = 3
    dataSetSize = dataSet.shape[0]
    # tile:平铺 沿某一方向复制
    # 这里的目的就是把每个坐标的差记录在diffMat中
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 这里的axis类似shape的维度
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# kNN.classify0([0,0],group,labels,3)


'''
    example - 2
'''


def file2matrix(filename):
    """提取文件到矩阵

    :param filename:
    :return:
    """
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    # get the number of lines in the file
    numberOfLines = len(arrayOLines)
    # prepare matrix to return
    returnMat = zeros((numberOfLines, 3))
    # prepare labels return
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 如果是数字直接写入否则根据love_dictionary转换
        if listFromLine[-1].isdigit():
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')

# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()

# ax = fig.add_subplot(349)
# 参数349的意思是：将画布分割成3行4列，图像画在从左到右从上到下的第9块
# 第十块，3410是不行的，可以用另一种方式(3,4,10)

# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# plt.show()


def autoNorm(dataSet):
    """归一化特征值

    :param dataSet:
    :return:归一化后的矩阵, 极差, 最小值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals
# normDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)


def datingClassTest():
    # hold out 10%
    hoRatio = 0.10
    # load data setfrom file
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is:{}".format(errorCount / float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    """
     每年获得的飞行常客里程数 28488
     玩视频游戏所耗时间百分比 10.528555
     每周消费的冰淇淋公升数 1.304844
    :return:
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: {}".format(resultList[classifierResult - 1]))


'''
    example - 3
'''


def img2vector(filename):
    """

    :param filename: 路径
    :return: 1 * 1024 矩阵
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')  # load the training set
    m = len(trainingFileList)
    # 训练数据拿出
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/{}'.format(fileNameStr))
    # iterate through the test set
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %{}".format(errorCount))
    print("\nthe total error rate is: {}".format(errorCount / float(mTest)))
