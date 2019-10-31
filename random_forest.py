import numpy as np
import decision_tree 
from decision_tree import build_tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tree import Tree
from import_data import datasets



def bootstrap(trainX_num):
    i=0
    result=np.array([],dtype='int32')
    while i<trainX_num:
        result=np.append(result,np.random.randint(0,trainX_num))
        i=i+1
    return result

def random_forest(trainX,trainy,trees_num,gr=0):
    i=0
    forest=[]
    while i<trees_num:
        trainset=bootstrap(len(trainX))
        #print(trainset)
        feature_num=trainX.shape[1]
        flags=[1]*feature_num
        K=round(np.log2(feature_num))
        
        subtree=Tree()
        temp_tree=build_tree(trainX[trainset],trainy[trainset],flags,depth=10,RF=1,K=K,gr=gr)
        subtree.data=temp_tree.data
        subtree.left=temp_tree.left
        subtree.right=temp_tree.right
        forest.append(subtree)
        print("第几棵树",i)
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


example_num=[500,1500,2500,3500,5000]
trees=[3]
test_scores1=[]
test_scores2=[]
test_scores3=[]


# trainX=dataX[:400]
# trainy=datay[:400]
# testX=dataX[400:600]
# testy=datay[400:600]

# forest=random_forest(trainX,trainy,5,gr=2)
# test1=score(testX,testy,forest)
# print(test1)
# for n in example_num:
for each in datasets:
    n=len(each[0])
    dataX=each[0]
    datay=each[1]
    folds=3
    fold_num=n//folds
    foldX=[dataX[:fold_num],dataX[fold_num:fold_num*2],dataX[fold_num*2:n]]
    foldy=[datay[:fold_num],datay[fold_num:fold_num*2],datay[fold_num*2:n]]

    for num in trees:
        i=0
        test1=0
        test2=0
        test3=0
        while i<folds:
            trainX=np.concatenate(foldX[0:i]+foldX[i+1:])
            trainy=np.concatenate(foldy[0:i]+foldy[i+1:])
            testX=foldX[i]
            testy=foldy[i]
            
            print("information gain\n\n\n")
            forest1=random_forest(trainX,trainy,num,gr=0)
            print("gain ratio\n\n\n\n")
            forest2=random_forest(trainX,trainy,num,gr=1)
            print("decision Stump\n\n\n")
            forest3=random_forest(trainX,trainy,num,gr=2)
            test1+=score(testX,testy,forest1)
            test2+=score(testX,testy,forest2)
            test3+=score(testX,testy,forest3)
            
            i+=1
        test_scores1.append(test1/folds)
        test_scores2.append(test2/folds)
        test_scores3.append(test3/folds)
    
    #trainX,testX,trainy,testy=train_test_split(dataX[:n],datay[:n],test_size=.4)
    
    
    
    
print(test_scores1,test_scores2,test_scores3)

plt1=plt.plot(list(range(6))[1:],test_scores1,linewidth=3,c='red',label='info gain')
plt2=plt.plot(list(range(6))[1:],test_scores2,linewidth=3,c='green',label='gain ratio')
plt3=plt.plot(list(range(6))[1:],test_scores3,linewidth=3,c='blue',label='decision stump')
plt.title('The relationship between different datasets and performance with 3 split methods')
plt.xlabel('Datasets ID')
plt.ylabel('Predictive accuracy')
plt.legend(loc='best')# make legend
plt.xlim(0,6)
plt.ylim(0.4,1)
plt.show()
