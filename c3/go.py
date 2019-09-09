
from c3.entropy import createTree,createPlot

with open('lenses.txt','r') as fw:
    content = fw.readlines()
    print(content)
    content = [ i.strip().split('\t') for i in content]
    print(content)
    labels = ['age','prescript','astigmatic','tearRate']
    tree = createTree(content,labels)

    createPlot(tree)