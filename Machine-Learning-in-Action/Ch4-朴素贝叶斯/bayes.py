from numpy import *
import re
import operator


'''
    example - 1
'''


# 加载数据
# 词表到向量的转换函数
def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性文字， 0代表正常言论
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


# 创建一个包含在所有文档中出现的不重复的列表
def create_vocab_list(data_set):
    # 创建一个空集
    vocab_set = set([])
    for document in data_set:
        # 创建两个集合的并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


# 输入参数是一个文档，输出的是文档向量
def set_of_word_vec(vocab_list, input_set):
    """该函数的输入参数为词汇表及某个文档, 输出的是文档向量,
    向量的每一元素为1或0, 分别表示词汇表中的单词在输入文档中是否出现。
    函数首先创建一个和词汇表等长的向量, 并将其元素都设置为0。接着，遍历文档中的所
    有单词，如果出现了词汇表中的单词,则将输出的文档向量中的对应值设为1。一切都顺利的话,
    就不需要检查某个词是否还在vocabList中，后边可能会用到这一操作。
    :param vocab_list:
    :param input_set:
    :return:
    """
    # 创建一个所含向量都为0的向量
    vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] = 1
        else:
            print("The word:%s is not in my vocabulary!" % word)
    return vec


# 朴素贝叶斯分类器的训练
def train(train_matrix, train_category):
    """利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，为防止为0概率
    可以将所有词的出现数初始化为1，并将分母初始化为2

    :param train_matrix: 文档矩阵
    :param train_category: 每篇文档类别标签所构成的向量
    :return: 侮辱性文档（class=1）的概率
    """
    # 矩阵中样本的数量
    num_train_docs = len(train_matrix)
    # 词汇表数量
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    # 初始化概率
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    # 向量相加
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # 对每个元素做除法
    # 下溢出，这是由于太多很小的数相乘造成的
    # p1_vec = p0_num / p1_denom
    # p0_vec = p0_num / p0_denom
    p1_vec = log(p1_num / p1_denom)
    p0_vec = log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive


# listOPosts, listClasses = load_data_set()
# myVocabList = create_vocab_list(listOPosts)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(set_of_word_vec(myVocabList, postinDoc))
# p0V, p1V, pAb = train(trainMat, listClasses)


# 朴素贝叶斯的分类
def classify(vec_classify, p0_vec, p1_vec, p_class):
    """先将两个向量中的第1个元素相乘，然后将第2个元素相乘，以此类推。
    接下来将词汇表中所有词的对应值相加，然后将该值加到类别的对数概率上。
    最后，比较类别的概率返回大概率对应的类别标签。

    :param vec_classify: 要分类的向量
    :param p0_vec:
    :param p1_vec:
    :param p_class:
    :return:
    """
    p1 = sum(vec_classify * p1_vec) + log(p_class)
    p0 = sum(vec_classify * p0_vec) + log(1 - p_class)
    if p1 > p0:
        return 1
    else:
        return 0


# 朴素贝叶斯的测试
def test():
    """测试函数

    """
    list_of_posts, list_classes = load_data_set()
    my_vocal_list = create_vocab_list(list_of_posts)
    train_matrix = []
    for post_in_doc in list_of_posts:
        train_matrix.append(set_of_word_vec(my_vocal_list, post_in_doc))
    p0_vec, p1_vec, p_abusive = train(array(train_matrix), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_word_vec(my_vocal_list, test_entry))
    print(test_entry, 'classified as:', classify(this_doc, p0_vec, p1_vec, p_abusive))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_word_vec(my_vocal_list, test_entry))
    print(test_entry, 'classified as: ', classify(this_doc, p0_vec, p1_vec, p_abusive))


'''
    example - 2 过滤垃圾邮件
'''


# 朴素贝叶斯的词袋模型
def bag_of_word_vec(vocab_list, input_set):
    """将每个词的出现与否作为一个特征，这可以被描述为词集模型如果一个词在文档中出现不止一次，
    这可能意味着包含该词是否出现在文档中所不能表达的某种信息，这种方法被称为词袋模型。

    :param vocab_list:
    :param input_set:
    :return:
    """
    vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] += 1
    return vec


# 切割分类文本
def text_parse(big_string):
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


# 垃圾邮件检测
def spam_test():
    """导入文件夹spam与ham下的文本文件，并将它们解析为词列表 。
    接下来构建一个测试集与一个训练集，两个集合中的邮件都是随机选出的。
    本例中共有50封电子邮件，并不是很多，其中的10封电子邮件被随机选择为测试集。
    分类器所需要的概率计算只利用训练集中的文档来完成。Python变量trainingSet
    是一个整数列表，其中的值从0到49。接下来，随机选择其中10个文件 。选择出的数字所对应
    的文档被添加到测试集，同时也将其从训练集中剔除。这种随机选择数据的一部分作为训练集，
    而剩余部分作为测试集的过程称为留存交叉验证（hold-out cross validation）。假定现在只完成了
    一次迭代，那么为了更精确地估计分类器的错误率，就应该进行多次迭代后求出平均错误率。
    """
    doc_list = []
    class_list = []
    full_text = []
    # 导入并且解析文本
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)

        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    train_set = list(range(50))
    test_set = []
    # 随机构建训练集
    for i in range(10):
        rand_index = int(random.uniform(1, len(train_set)))
        test_set.append(train_set[rand_index])
        del (train_set[rand_index])
    train_matrix = []
    train_class = []
    # 对测试集合进行分类
    for doc_index in train_set:
        train_matrix.append(bag_of_word_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
        p0_vec, p1_vec, p_spam = train(array(train_matrix), array(train_class))
        error_count = 0
        for doc_index in test_set:
            word_vector = bag_of_word_vec(vocab_list, doc_list[doc_index])
            if classify(array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
                error_count += 1
                # print("classification error", doc_list[doc_index])
        print('the error rate is: ', float(error_count) / len(test_set))


# RSS源分类器和高频去除函数
def calc_most_freq(vocab_list, full_text):
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)

    sorted_freq = sorted(freq_dict.tems(),
                         key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed0, feed1):
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    # 每次访问一条RSS源
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)  # NY is class 1
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)

    # 去掉那一些出现频率最高的词
    top_words = calc_most_freq(vocab_list, full_text)
    for pair_w in top_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])
    train_set = range(2 * min_len)
    test_set = []  #
    for i in range(20):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del (train_set[rand_index])
    train_matrix = []
    train_class = []
    for docIndex in train_set:
        train_matrix.append(bag_of_word_vec(vocab_list, doc_list[docIndex]))
        train_class.append(class_list[docIndex])
    p0_vec, p1_vec, p_spam = train(array(train_matrix), array(train_class))
    error_count = 0
    for doc_index in test_set:
        word_vector = bag_of_word_vec(vocab_list, doc_list[doc_index])
        if classify(array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            error_count += 1
    print('the error rate is: ', float(error_count) / len(test_set))
    return vocab_list, p0_vec, p1_vec


# 最具特征性的词汇显示函数
def get_top_words(ny, sf):
    vocab_list, p0_vec, p1_vec = local_words(ny, sf)
    top_ny = []
    top_sf = []

    for i in range(len(p0_vec)):
        if p0_vec[i] > -6.0:
            top_sf.append((vocab_list[i], p0_vec[i]))
        if p1_vec[i] > -6.0:
            top_ny.append((vocab_list[i], p1_vec[i]))
    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    print("sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**")
    for item in sorted_sf:
        print(item[0])
    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print("ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**")
    for item in sorted_ny:
        print(item[0])
