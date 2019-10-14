import graphviz
from sklearn import datasets
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

cancer=datasets.load_breast_cancer()
clf=tree.DecisionTreeClassifier(max_depth=10)
dataX=cancer.data
datay=cancer.target
example_num=[5,100,200,300,400,500]
train_scores=[]
test_scores=[]

for n in example_num:
    np.random.seed(10)
    indices=np.random.permutation(len(dataX))
    #train_num=n*2//3
    train_num=n
    trainX=dataX[indices[:train_num]]
    trainy=datay[indices[:train_num]]
    testX=dataX[indices[train_num:]]
    testy=datay[indices[train_num:]]
    clf=clf.fit(trainX,trainy)
    train_scores.append(clf.score(trainX,trainy))
    test_scores.append(clf.score(testX,testy))

plt.plot(example_num,train_scores,linewidth=3,c='red')
plt.plot(example_num,test_scores,linewidth=3,c='green')
plt.show()

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph=graphviz.Source(dot_data)
# graph.render("iris")
