"""
    example - 1 Logistic 回归梯度上升优化算法
"""

import matplotlib.pyplot as plt
from numpy import *
from numpy.ma import array


# 加载数据 两个特征值
def loadDataSet():
    dataMatrix = []
    labelMatrix = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_array = line.strip().split()
        dataMatrix.append([1.0, float(line_array[0]), float(line_array[1])])
        labelMatrix.append(int(line_array[2]))
    return dataMatrix, labelMatrix


def sigmoid(inX):
    """numpy数组x中可能有绝对值比较大的负数，这样传给sigmoid函数时，分母exp(-x)会非常大，导致 exp(-x) 溢出

    :param inX:
    :return:
    """
    # return (1.0 / (1+math.exp(-inX))) + 1e-8
    # return 1.0 / (1.0 + exp(-inX))
    x_ravel = inX.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + exp(-x_ravel[index])))
        else:
            y.append(exp(x_ravel[index]) / (exp(x_ravel[index]) + 1))
    return array(y).reshape(inX.shape)


# Logistic回归梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    # 将数据转化为numpy矩阵
    dataMat = mat(dataMatIn)
    # 为了便于矩阵运算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)

    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))

    # 矩阵之间做乘法
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error

    return weights


# DataMatrix, labelMatrix = loadDataSet()
# print(gradAscent(DataMatrix, labelMatrix))


# 画图
def plotBestFit(weights):
    data_matrix, label_matrix = loadDataSet()
    data_array = array(data_matrix)
    n = shape(data_array)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(label_matrix[i]) == 1:
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)

    # 最佳似合直线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


DataMatrix, labelMatrix = loadDataSet()


# print(gradAscent(DataMatrix, labelMatrix))
# plotBestFit(gradAscent(DataMatrix, labelMatrix))


# 随机梯度上升算法
def stocGradAscent0(data_matrix, class_labels):
    """随机梯度上升算法与梯度上升算法在代码上很相似

    :param data_matrix:
    :param class_labels:
    :return:
    """
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * array(data_matrix[i])

    return weights


# DataMatrix, labelMatrix = loadDataSet()
# print(stocGradAscent0(array(DataMatrix), labelMatrix))
# plotBestFit(stocGradAscent0(DataMatrix, labelMatrix))


# 改进后的随机梯度上升算法
def stocGradAscent1(data_matrix, class_labels, num_iter=150):
    """每次来调整alpha的值
    通过随机选取样本来更新回归系数

    :param data_matrix:
    :param class_labels:
    :param num_iter:
    :return:
    """
    m, n = shape(data_matrix)
    weights = ones(n)

    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            # 每次来调整alpha的值虽然alpha会随着迭代次数不断减小，但永远不会减小到0，这是因为还存在一个常数项。
            alpha = 4 / (1.0 + j + i) + 0.0001
            # 通过随机选取样本来更新回归系数。这种方法将减少周期性的波动。具体实现方法与第3章类似，这种方法每次随机从列表中选出一个值，然后从列表中删掉该值（再进行下一次迭代）。
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * array(data_matrix[rand_index])
            del (data_index[rand_index])
    return weights


# plotBestFit(stocGradAscent1(DataMatrix, labelMatrix))
'''
    example - 2 从疝气病症预测病马的死亡率
    
    实数0来替换所有缺失值
'''


# Logistic回归分类函数
def classify_vector(inx, weights):
    """

    :param inx: 特征值
    :param weights: 计算出的权重
    :return:
    """
    prob = sigmoid(sum(inx * weights))

    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    frtrain = open('horseColicTraining.txt');
    frtest = open('horseColicTest.txt')
    train_set = []
    train_labels = []

    for line in frtrain.readlines():
        curr_line = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(curr_line[i]))
        train_set.append(line_array)
        train_labels.append(float(curr_line[21]))

    train_weights = stocGradAscent1(array(train_set), train_labels, 1000)
    error_count = 0
    num_test_vector = 0.0

    for line in frtest.readlines():
        num_test_vector += 1.0
        curr_line = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(curr_line[i]))
        if int(classify_vector(array(line_array), train_weights)) != int(curr_line[21]):
            error_count += 1

    error_rate = (float(error_count) / num_test_vector)
    print(f"the error rate of this test is: {error_rate}")

    return error_rate


# 多重测试
# after 10 iterations the average error rate is: 0.34029850746268653
def mul_test():
    num_tests = 10
    error_sum = 0.0

    for k in range(num_tests):
        error_sum += colic_test()

    print(f"after {num_tests} iterations the average error rate is: {error_sum / float(num_tests)}")


mul_test()
