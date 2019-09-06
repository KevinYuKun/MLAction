from collections import Counter
from math import log2
import numpy as np
import copy
import operator

'''
    计算根节点信息熵(总体分类结果)
    return base entropy, 总体的分类label
'''
def calEnt(dataset):
    len_dataset = len(dataset)
    label_count = {}
    # 得到根节点总分类结果
    for feature in dataset:
        currentLabel = feature[-1]
        if currentLabel not in label_count:
            label_count[currentLabel] = 0
        label_count[currentLabel] += 1

    # 计算根节点信息熵
    entropy = 0.0
    for  key,value in label_count.items():
        prob = float(value/len_dataset)
        entropy += (-prob)*log2(prob)

    # 返回总分类个数
    return entropy,label_count.keys()

'''
    计算信息增益
    return 信息增益最大的feature的index，和信息增益值
'''
def cal_information_gain(dataset,entropy,labels):
    num_of_feature = len(dataset[0]) - 1
    best_info_gain = 0.0
    best_feature = -1
    num_of_data = len(dataset)

    # 遍历每个feature
    # 双层嵌套字典
    label_list = {}
    # 初始化每个label
    for x in labels:
        label_list[x] = 0
    for feature in range(num_of_feature):

        feature_list = {}


        # 对于每条数据，添加嵌套字典
        for data in dataset:
            current_label = data[-1]
            if data[feature] not in feature_list:

                feature_list[data[feature]] = copy.copy(label_list)

            feature_list[data[feature]][current_label] += 1

        # 计算条件熵
        con_ent = 0.0
        for x in feature_list:
            # print(feature_list[x].items())
            total_num = np.sum(list(feature_list[x].values()))
            temp_ent = 0.0
            for y,num in feature_list[x].items():
                if num != 0:
                    rate = float(num / total_num)
                    temp_ent +=  -rate * log2(rate)


            con_ent += float(total_num/num_of_data)*temp_ent

        if (entropy - con_ent) > best_info_gain:
            best_info_gain = (entropy - con_ent)
            best_feature = feature


    return best_feature,best_info_gain


'''
    返回该子节点中分类出现次数最多的一个
'''
def Max(classList):
    result = dict(Counter(classList))
    # 根据value排序
    result = sorted(result.items(),key=operator.itemgetter(1),reverse=True)

    return result[0][0]



'''
    构建决策树 labels为所有feature
    递归结束条件：
        1. 所有分类class都相同，在这里是全为no或yes
        2. 所有特征均已使用
'''
def createTree(dataset,labels):
    classList = [example[-1] for example in dataset]
    # 条件1 判断出现是否一致
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 条件2 所有特征都已使用，只剩label，选择出现次数最多的
    if len(dataset[0]) == 1:
        return Max(classList)

    tmp_ent, tmp_label = calEnt(dataset)
    best_info_gain_index = cal_information_gain(dataset,tmp_ent,tmp_label)[0]
    best_feature = labels[best_info_gain_index]

    myTree = {best_feature:{}}
    # 用完就删特征
    del(labels[best_info_gain_index])

    # 获取当前feature的所有值
    feature_value = [x[best_info_gain_index] for x in dataset]
    uni_feature_value = set(feature_value)

    #  进行分叉
    for value in uni_feature_value:
        subLabels = labels[:]

        splData = []
        # 分离其他feature
        for data in dataset:
            if data[best_info_gain_index] == value:
                reduce_data = data[0:best_info_gain_index]
                reduce_data.extend(data[best_info_gain_index+1:])
                splData.append(reduce_data)

        myTree[best_feature][value] = createTree(splData,subLabels)

    return myTree


''' 
test data
'''
dataset  = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
feature_labels = ['no surfacing','flippers']
# print(calEnt(dataset))
# ent,lab = calEnt(dataset)
# print(cal_information_gain(dataset,ent,lab))
print(createTree(dataset,feature_labels))