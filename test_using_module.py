from sklearn import datasets
import numpy as np
from numpy import mat,matrix
import math
from sklearn.datasets.samples_generator import make_classification
from gradient_decent import gd_logistic_regression,score,predict,model_func


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








# para_w=mat(np.array([1.0,1.0,-1.0,-1.0]))
# print(para_w)
# para_t=0
# learning_rate=0.1
# maxepochs=300
# epoch=0


# while epoch<maxepochs :
#     j=0
#     for x,y in zip(trainX,trainy):
#         temp_w=para_w
#         while j<dimension:
#             para_w[0,j]=para_w[0,j]-learning_rate*(model_func(x,para_w,para_t)-y)*x[j]
#             j=j+1
#         para_t=para_t+learning_rate*(model_func(x,temp_w,para_t)-y)
#     epoch=epoch+1
para_w,para_t=gd_logistic_regression(trainX,trainy,maxepochs=300,learning_rate=0.22)
score(para_w,para_t)
