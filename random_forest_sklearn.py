from sklearn.ensemble import RandomForestClassifier
from import_data import dataX,datay
import numpy as np
import matplotlib.pyplot as plt

example_num=[500]
trees=[3,5]
test_score=[]



def score(testX,testy,clf):
    i=0
    right=0
    while i<len(testX):
        temp=clf.predict(testX)
        if temp[0]==testy[i]:
            right+=1
        i+=1
    return (right/len(testX))

for n in example_num:

    folds=3
    fold_num=n//folds
    foldX=[dataX[:fold_num],dataX[fold_num:fold_num*2],dataX[fold_num*2:n]]
    foldy=[datay[:fold_num],datay[fold_num:fold_num*2],datay[fold_num*2:n]]

    for num in trees:
        i=0
        test1=0
        test2=0
        while i<folds:
            trainX=np.concatenate(foldX[0:i]+foldX[i+1:])
            trainy=np.concatenate(foldy[0:i]+foldy[i+1:])
            testX=foldX[i]
            testy=foldy[i]
            
            clf=RandomForestClassifier(max_depth=15,n_estimators=n)
            clf.fit(trainX,trainy)
            test1+=score(trainX,trainy,clf)
            
            i+=1
        test_score.append(test1/folds)


print(test_score)
plt1=plt.plot(trees,test_score,linewidth=3,c='red',label='info gain')
plt2=plt.plot(trees,test_score,linewidth=3,c='green',label='gain ratio')
plt.title('The relationship between the number of trees and performance')
plt.xlabel('Number of trees in Forest')
plt.ylabel('Predictive accuracy')
plt.legend(loc='best')# make legend
# plt.xlim(0,7)
plt.ylim(0,1)
plt.show()