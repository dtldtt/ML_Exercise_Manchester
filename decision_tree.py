import numpy as np 
from sklearn import datasets
from decision_stump import decision_stump
from sklearn import metrics
from tree import Tree
import warnings
warnings.filterwarnings("ignore")


cancer=datasets.load_breast_cancer()
dataX=cancer.data
datay=cancer.target
example_num=[5,100,200,300,400,500]
train_scores=[]
test_scores=[]




def build_tree(trainX,trainy,tree=Tree(),depth=10):
    if depth==0 or len(np.unique(trainy))==1:
        count0=list(trainy).count(0)
        count1=list(trainy).count(1)
        if count0>count1:
            return 0
        else:
            return 1
    i=0
    feature_score=0
    best_feature=0
    while i<(trainX.shape[1]):
        # try split the first feature
        #decision_stump(trainX.T[i],trainy)
        current_feature=metrics.adjusted_mutual_info_score(trainX.T[i],trainy)
        if current_feature>feature_score:
            feature_score=current_feature
            best_feature=i
        i=i+1
    #print(trainX.T[best_feature],trainy)
    threshold=decision_stump(trainX.T[best_feature],trainy)
    #print(threshold)
    tree.data=[best_feature,threshold]
    left_tree=Tree()
    right_tree=Tree()
    leftsub=np.array([],dtype='int32')
    rightsub=np.array([],dtype='int32')
    
    # split into 2 parts: leftsub and rightsub
    i=0
    while i<(trainX.shape[0]):
        if trainX[i,best_feature]<=threshold:
            leftsub=np.append(leftsub,i)
        else:
            rightsub=np.append(rightsub,i)
        i=i+1
    #print(leftsub,rightsub)
    if len(leftsub)>0:
        left_tree=build_tree(trainX[leftsub],trainy[leftsub],tree=left_tree,depth=depth-1)
        tree.left=left_tree
    if len(rightsub)>0:
        right_tree=build_tree(trainX[rightsub],trainy[rightsub],tree=right_tree,depth=depth-1)
        tree.right=right_tree
    
    return tree

def predict(testX,tree):
    current_tree=tree
    while type(current_tree)==Tree:
        if testX[current_tree.data[0]]<current_tree.data[1]:
            current_tree=current_tree.left
        else:
            current_tree=current_tree.right
    return current_tree

def score(testX,testy,tree):
    right=0
    for i in range(len(testX)):
        predict_y=predict(testX[i],tree)
        if predict_y==testy[i]:
            right=right+1
    return round(right/len(testX),2)

for n in example_num:
    np.random.seed(10)
    indices=np.random.permutation(len(dataX))
    #train_num=n*2//3
    train_num=n
    trainX=dataX[indices[:train_num]]
    trainy=datay[indices[:train_num]]
    testX=dataX[indices[train_num:]]
    testy=datay[indices[train_num:]]
    tree=build_tree(trainX,trainy,depth=4)
    train_scores.append(score(trainX,trainy,tree))
    test_scores.append(score(testX,testy,tree))
print(train_scores,test_scores)
