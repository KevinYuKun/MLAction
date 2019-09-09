
'''
    测试数据
'''
def classify(inputTree, feature, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # print(feature)
    featureIndex = feature.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], feature, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel