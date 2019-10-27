import numpy as np
import decision_tree 
from decision_tree import build_tree,dataX,datay
import matplotlib.pyplot as plt
from sklearn import metrics
from tree import Tree

def bootstrap(trainX_num):
    i=0
    result=np.array([],dtype='int32')
    while i<trainX_num:
        result=np.append(result,np.random.randint(0,trainX_num))
        i=i+1
    return result

def random_forest(trainX,trainy,trees_num):
    i=0
    forest=[]
    while i<trees_num:
        trainset=bootstrap(len(trainX))
        #print(trainset)
        feature_num=trainX.shape[1]
        flags=[1]*feature_num
        K=round(np.log2(feature_num))+1
        
        subtree=Tree()
        temp_tree=build_tree(trainX[trainset],trainy[trainset],flags,depth=5,RF=1,K=K)
        subtree.data=temp_tree.data
        subtree.left=temp_tree.left
        subtree.right=temp_tree.right
        forest.append(subtree)
        i=i+1
    return forest

def predict(testX,forest):
    result=[]
    for tree in forest:
        result.append(decision_tree.predict(testX,tree))
    count0=result.count(0)
    count1=result.count(1)
    #print(result)
    if count0>count1:
        return 0
    else:
        return 1

def score(testX,testy,forest):
    right=0
    for i in range(len(testX)):
        result=predict(testX[i],forest)
        if result==testy[i]:
            right+=1
    return round(right/len(testX),2)


example_num=[50,200,400]
train_scores=[]
test_scores=[]

for n in example_num:
    np.random.seed()
    indices=np.random.permutation(len(dataX))
    #train_num=n*2//3
    train_num=n
    # trainX=dataX[indices[:train_num]]
    # trainy=datay[indices[:train_num]]
    # testX=dataX[indices[train_num:]]
    # testy=datay[indices[train_num:]]
    trainX=dataX[:n]
    trainy=datay[:n]
    testX=dataX[n:]
    testy=datay[n:]
    forest=random_forest(trainX,trainy,35)
    train_scores.append(score(trainX,trainy,forest))
    test_scores.append(score(testX,testy,forest))
print(train_scores,test_scores)

plt.plot(example_num,train_scores,linewidth=3,c='red')
plt.plot(example_num,test_scores,linewidth=3,c='green')
plt.show()
