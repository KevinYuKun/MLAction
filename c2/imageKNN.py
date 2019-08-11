import os
import numpy as np
from c2.KNN import classify
test_path = "/Users/icream/PycharmProjects/ML/machinelearninginaction/Ch02/digits/testDigits"
train_path = "/Users/icream/PycharmProjects/ML/machinelearninginaction/Ch02/digits/trainingDigits"
testNum = len(os.listdir(test_path))
trainNum = len(os.listdir(train_path))

# print(testNum,trainNum)

def img2vector(fileName):
    fr = open(fileName)
    content = fr.readlines()
    returnVector = np.zeros([1,1024])

    for i in range(len(content)):
        line = content[i]
        line = line.strip("\n")
        # print()
        for k in range(len(line)):
            returnVector[0,32*i+k] = line[k]
    # print(returnVector)
    return returnVector

def HWclassify():
    train_hw_label = []
    train_hw_data  = np.zeros([trainNum,1024])
    # print(trainNum)
    for i in range(len(os.listdir(train_path))):
        fileName = os.listdir(train_path)[i]

        # get the label
        train_hw_label.append(int(fileName.split("_")[0]))
        # print(fileName)

        # get the data
        train_hw_data[i] = img2vector(train_path+"/"+fileName)

    errCount = 0
    for i in range(len(os.listdir(test_path))):
        fileName = os.listdir(test_path)[i]
        test_label = int(fileName.split("_")[0])
        test_data = img2vector(test_path+"/"+fileName)



        result = classify(test_data,train_hw_data,train_hw_label,3)
        print("test label: {}     return answer : {}".format(test_label,result))

        if (int(result) != test_label):
            errCount += 1


    print("err rate: ",errCount/testNum)



if __name__ == '__main__':
    HWclassify()