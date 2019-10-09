import numpy as np
from numpy import mat,matrix
import math
from gradient_decent import gd_logistic_regression,model_func,predict,score,trainX,trainy


trainX_fold1=trainX[:100]
trainX_fold2=trainX[100:200]
trainX_fold3=trainX[200:300]
trainX_fold4=trainX[300:400]
testX_fold=trainX[400:]
#make the first 4 Xfolds into the trainX set
trainX=[]
trainX.append(trainX_fold1)
trainX.append(trainX_fold2)
trainX.append(trainX_fold3)
trainX.append(trainX_fold4)


trainy_fold1=trainy[:100]
trainy_fold2=trainy[100:200]
trainy_fold3=trainy[200:300]
trainy_fold4=trainy[300:400]
testy_fold=trainy[400:]
#make the first yfolds into the trainy set
trainy=[]
trainy.append(trainy_fold1)
trainy.append(trainy_fold2)
trainy.append(trainy_fold3)
trainy.append(trainy_fold4)

final_score=0
crosses=0
flag=0      #which one is valid fold, other 3 are train fold
result=0

while crosses<len(trainX):
    w=mat(np.random.randn(1,trainX_fold1.shape[1]))
    t=0

    for x,y in zip(trainX,trainy):
        if (x==trainX[crosses]).all():
            continue
        w,t=gd_logistic_regression(x,y,para_t=t,para_w=w,learning_rate=0.2,maxepochs=100)

    temp_score=score(w,t,trainX[crosses],trainy[crosses])
    if temp_score>final_score:
        final_score=temp_score
        flag=crosses
        print(final_score,flag)
        result=score(w,t,testX_fold,testy_fold)    
    crosses=crosses+1

print(result)


