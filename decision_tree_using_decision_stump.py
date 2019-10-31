import numpy as np 
from sklearn import datasets
from decision_stump import decision_stump
from sklearn import metrics
from tree import Tree
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")






def build_tree(trainX,trainy,flags,tree=Tree(),depth=10):
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

    #flags=[1]*(trainX.shape[1])
    
    while i<(trainX.shape[1]):
        if flags[i]==0:
            i=i+1
            continue
        train_feature=np.array([],dtype='int32')
        
        feature_divides=np.percentile(trainX.T[i],(25,50,75),interpolation='midpoint')
    
        for eachX in trainX.T[i]:
            if eachX<feature_divides[0]:
                train_feature=np.append(train_feature,1)
            elif eachX<feature_divides[1]:
                train_feature=np.append(train_feature,2)
            elif eachX<feature_divides[2]:
                train_feature=np.append(train_feature,3)
            else:
                train_feature=np.append(train_feature,4)
        #print(len(trainX.T[i]),len(train_feature),len(trainy))
        current_feature=metrics.mutual_info_score(train_feature,trainy)
        if current_feature>feature_score:
            feature_score=current_feature
            best_feature=i
        i=i+1

     
    #print(trainX.T[best_feature],trainy)
    flags[best_feature]=0
    #print(depth)
    #print(flags)   
    threshold=decision_stump(trainX.T[best_feature],trainy,step_size=0.1)
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
        left_tree=build_tree(trainX[leftsub],trainy[leftsub],flags,tree=left_tree,depth=depth-1)
        tree.left=left_tree
    if len(rightsub)>0:
        right_tree=build_tree(trainX[rightsub],trainy[rightsub],flags,tree=right_tree,depth=depth-1)
        tree.right=right_tree
    flags[best_feature]=1
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
    py_set=[]
    for i in range(len(testX)):
        predict_y=predict(testX[i],tree)
        py_set.append(predict_y)
        if predict_y==testy[i]:
            right=right+1
    #print(py_set)
    #print(testy)
    return round(right/len(testX),2)

cancer=datasets.load_breast_cancer()
dataX=cancer.data
datay=cancer.target
example_num=[550]
test_scores=[]


for n in example_num:
    #np.random.seed()
   

    folds=3
    fold_num=n//folds
    foldX=[dataX[:fold_num],dataX[fold_num:fold_num*2],dataX[fold_num*2:n]]
    foldy=[datay[:fold_num],datay[fold_num:fold_num*2],datay[fold_num*2:n]]
    depths=[5,8,10,15]
    for d in depths:
        i=0
        test1=0
        
        while i<folds:
            trainX=np.concatenate(foldX[0:i]+foldX[i+1:])
            trainy=np.concatenate(foldy[0:i]+foldy[i+1:])
            testX=foldX[i]
            testy=foldy[i]
  
            flags=[1]*(trainX.shape[1])
            
            tree=build_tree(trainX,trainy,flags,depth=d)
            test1+=score(testX,testy,tree)

            i+=1
        test_scores.append(test1/folds)
    
    #trainX,testX,trainy,testy=train_test_split(dataX[:n],datay[:n],test_size=.4)
    
    
    
    
print(test_scores)

plt1=plt.plot(depths,test_scores,linewidth=3,c='red')

plt.title('The relationship between the depth of tree and performance')
plt.xlabel('The depth of tree')
plt.ylabel('Predictive accuracy')
plt.legend(loc='best')# make legend
# plt.xlim(0,7)
plt.ylim(0,1)
plt.show()
