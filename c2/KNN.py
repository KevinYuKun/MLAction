import numpy as np

import matplotlib.pyplot as plt
import operator




dataSet = np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
labels = np.array(["A","A","B","B"])
tr_labels = np.array([["A"],['A'],['B'],['B']])

# inputX 单个点
# 训练得出结果
def classify(inputX, train_set,labels,k):
    # if(train_set.shape[1] != len(inputX[0])):
    #         print("vector not same")
    #         exit(1)

    train_row = train_set.shape[0]
    # 点到点距离
    diffMat = np.tile(inputX,(train_row,1)) - train_set

    # 欧式距离(直线距离)----
    sqDiffMat = diffMat ** 2
    # x + y
    Distance = (np.sum(sqDiffMat,axis=1))**0.5
    sortIndex = np.argsort(Distance)

    # print(sortIndex[3])
    # 投票
    classCount={}
    for i in range(k):
        voteLabel = labels[sortIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    # 排序
    sortedCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)

    return sortedCount[0][0]

# 将txt转换成合适的格式
def file2Matrix(file):
    fr = open(file)
    content_lines = fr. readlines()
    num_lines = len(content_lines)

    #  创建一个1000X3 的0矩阵
    mat = np.zeros((num_lines,3))
    labels = np.array([])

    flag = 0
    for line in content_lines:
        # 对每一行进行划分
        line = line.strip()

        temp_data_list = line.split('\t')
        # 替换 0 矩阵
        mat[flag,:] = temp_data_list[0:3]
        labels = np.append(labels,[int(temp_data_list[-1])])

        flag += 1
    return mat,labels


def autoNorm(dataSet):
    # 归一化
    minVals  = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normSet =dataSet - np.tile(minVals,(np.shape(dataSet)[0],1))
    normSet /= np.tile(ranges, (np.shape(dataSet)[0],1))
    return normSet

def datingClassTestd():
    testRatio = 0.1
    dataSet,labels = file2Matrix("datingTestSet2.txt")
    norm = autoNorm(dataSet)
    errorCount = 0
    m = np.shape(norm)[0]
    numTest = int(testRatio*m)

    for i in range(numTest):
        classResult = classify(norm[i,:],norm[numTest:m,:],labels[numTest:m],3)
        # print(int(classResult))
        if(int(classResult) != labels[i]):
            errorCount += 1

    print("error rate :", errorCount/numTest)

if __name__ == '__main__':
    fileName = "datingTestSet2.txt"
    train_data,train_label =file2Matrix(fileName)

    # 画图
    fig = plt.figure()
    # 子图 234-----> 2x3网格 第4子图
    ax = fig.add_subplot(111)
    # test = np.array([[1,3,5],[2,0,4],[9,2,9]])
    # print(train_data.max(axis=0))
    # print(train_data)
    # 画多个标签legend
    # 取 0 1两列的点
    pt1_x, pt1_y = [],[]
    pt2_x, pt2_y = [],[]
    pt3_x, pt3_y = [],[]

    # print(train_data[2,0:3])

    norm = autoNorm(train_data)

    for i in range(np.shape(norm)[0]):
        if(train_label[i] == 1):
            pt1_x.append(norm[i,0])
            pt1_y.append(norm[i,1])
        elif train_label[i] == 2:
            pt2_x.append(norm[i, 0])
            pt2_y.append(norm[i, 1])
        else:
            pt3_x.append(norm[i, 0])
            pt3_y.append(norm[i, 1])


    # ax.scatter(train_data[:,0],train_data[:,1],
    #            120.0,np.array(train_label),"x")
    t1 = ax.scatter(pt1_x,pt1_y,12,c='red',marker = "o")
    t2 = ax.scatter(pt2_x,pt2_y,12,c='blue',marker = "x")
    t3 = ax.scatter(pt3_x,pt3_y,12,c='green',marker = "*")
    plt.legend([t1,t2,t3],["dogs","honest","father"])
    plt.ylabel("%")
    plt.xlabel("numbers")
    # plt.legend('x1')
    plt.show()
    datingClassTestd()

    # 定义测试例子
    # tst = np.array(([0.1,0.2,0.3],[1,2,3]))

    # 分隔测试集为单个
    # try:
    #     new_tst = np.vsplit(tst, tst.shape[0])
    # except:
    #         print("vector error")
    #         exit(1)
    # for x in range(len(new_tst)):
    #     print("{} is {}".format(new_tst[x],classify(new_tst[x],train_data,train_label,3)))