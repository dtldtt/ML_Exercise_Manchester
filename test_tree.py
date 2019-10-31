import numpy as np 
from sklearn import datasets
from decision_stump import decision_stump
from sklearn import metrics
from sklearn import tree

import warnings
import matplotlib.pyplot as plt
#from import_data import dataX,datay
warnings.filterwarnings("ignore")


# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(dataX[:5000], datay[:5000])
# print(clf.score(dataX[5000:7000],datay[5000:7000]))

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
            
            clf = tree.DecisionTreeClassifier(max_depth=d)
            clf = clf.fit(trainX, trainy)
            
            test1+=clf.score(testX,testy)

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