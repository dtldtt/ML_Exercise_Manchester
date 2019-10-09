from sklearn import datasets
import numpy as np
from numpy import mat,matrix
import math
from sklearn.datasets.samples_generator import make_classification


iris=datasets.load_iris()
irisX=iris.data
iris_y=iris.target

#data=datasets.make_hastie_10_2()
#dataX=data[0]
#datay=data[1]

# trainX=np.concatenate((irisX[:40],irisX[50:90]),axis=0)
# trainy=np.concatenate((iris_y[:40],iris_y[50:90]),axis=0)
# testX=np.concatenate((irisX[40:50],irisX[90:100]),axis=0)
# testy=np.concatenate((iris_y[40:50],iris_y[90:100]),axis=0)

#for i in range(len(datay)):
#    if datay[i]==-1:
#        datay[i]=0

dataX,datay=make_classification(n_samples=500,n_features=4,n_redundant=0,n_informative=2,
                             random_state=1,n_clusters_per_class=2)

trainX=dataX[:400]
trainy=datay[:400]
testX=dataX[400:]
testy=datay[400:]
dimension=4

def model_func(x,w,t):
    return 1.0/(1+math.exp(-(x*w.T-t).mean()))

def predict(w,t):
    predict_y=[]
    for each_x in testX:
        predict_y.append(round(model_func(each_x,w,t)))
    print(predict_y)
    print(testy)
    return predict_y


def score(w,t):
    predict_y=predict(w,t)
    right=0
    for i in range(len(predict_y)):
        if predict_y[i]==testy[i]:
            right=right+1
    print(round(right/len(predict_y),2))

# para_w=mat(np.array([1.0,1.0,-1.0,-1.0]))
# print(para_w)
# para_t=0
# learning_rate=0.1
# maxepochs=300
# epoch=0

def gd_logistic_regression(trainX,trainy,dimension=trainX.shape[1],maxepochs=200,
                           learning_rate=0.1,para_t=1,para_w=mat(np.random.randn(1,dimension))):
    epoch=0
    while epoch<maxepochs:
        j=0
        for x,y in zip(trainX,trainy):
            temp_w=para_w
            while j<dimension:
                para_w[0,j]=para_w[0,j]-learning_rate*(model_func(x,para_w,para_t)-y)*x[j]
                j=j+1
            para_t=para_t+learning_rate*(model_func(x,temp_w,para_t)-y)
        epoch=epoch+1
    return para_w,para_t

# while epoch<maxepochs :
#     j=0
#     for x,y in zip(trainX,trainy):
#         temp_w=para_w
#         while j<dimension:
#             para_w[0,j]=para_w[0,j]-learning_rate*(model_func(x,para_w,para_t)-y)*x[j]
#             j=j+1
#         para_t=para_t+learning_rate*(model_func(x,temp_w,para_t)-y)
#     epoch=epoch+1
# para_w,para_t=gd_logistic_regression(trainX,trainy,maxepochs=300,learning_rate=0.2)
# score(para_w,para_t)
