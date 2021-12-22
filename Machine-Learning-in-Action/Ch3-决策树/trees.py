import math
import operator
import pickle


def create_dataset():
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# dataSet, labels=trees.create_dataset()


def calcShannonEnt(dataSet):
    """计算香农熵, 得到熵之后, 我们就可以按照获取最大信息增益的方法划分数据集,
    另一个度量集合无序程度的方法是基尼不纯度

    :param dataSet:
    :return: 香农熵
    """
    num_entries = len(dataSet)
    label_counts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        # 拿到最后一个为label
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannoEnt = 0.0

    # 以二为底求对数
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannoEnt -= prob * math.log(prob, 2)
    return shannoEnt


def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集

    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    """
    # 创建新的list对象
    retdataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retdataSet.append(reducedFeatVec)
    return retdataSet


def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分方式
    第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；
    第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。数据集一旦满足上述要求，我们就可以在函数的第一行判定当前数据集包含多少特征属性。
        我们无需限定list中的数据类型，它们既可以是数字也可以是字符串，并不影响实际计算。

    :param dataSet:
    :return: 最好特征的索引
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 创建唯一分类标签
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计划每种划分的信息墒
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的增益墒
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """该函数使用分类名称的列表,然后创建键值为classList中唯一值的数据字典,字典对象存储了classList中每个类标签出现的频率,
    最后利用operator操作键值排序字典,并返回出现次数最多的分类名称。

    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def create_tree(dataSet, labels):
    """创建树的函数代码

    :param dataSet:
    :param labels:
    :return:
    """
    # ['yes', 'yes', 'no', 'no', 'no']
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 停止分类直至所有类别相等
        return classList[0]
    if len(dataSet[0]) == 1:
        # 停止分割直至没有更多特征
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到包含所有属性的列表
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = create_tree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    # 写入为字节
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    # 文件模式为字节处理
    fr = open(filename, "rb+")
    return pickle.load(fr)

# fr = open('lenses.txt')
# lenses=[inst.strip().split('\t') for inst in fr.readlines()]
# lensesLabels=['age','prescript','astigmatic','tearRate']
# lensesTree=trees.create_tree(lenses,lensesLabels)
# treePlotter.createPlot(lens)

