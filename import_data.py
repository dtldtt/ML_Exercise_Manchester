import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import datasets


#----------------------------------------------------------------

tmp = np.loadtxt("default of credit card clients.csv", dtype=np.str, delimiter=",")
dataX_credit = tmp[2:,1:-1].astype(np.float)#加载数据部分
datay_credit = tmp[2:,-1].astype(np.int32)#加载类别标签部分

dataX_credit=dataX_credit[:1000]
datay_credit=datay_credit[:1000]
#print(data,label)

#------------------------------------------------------------------

tmp = np.loadtxt("echocardiogram.data", dtype=np.str, delimiter=",")
dataX_echocardiogram = tmp[:,0:-2].astype(np.float)#加载数据部分S
datay_echocardiogram = tmp[:,-1].astype(np.int32)#加载类别标签部分
i=0
flag=[]
while i<len(dataX_echocardiogram):
    if datay_echocardiogram[i]==-1:
        flag.append(i)
    i+=1
dataX_echocardiogram=np.delete(dataX_echocardiogram,flag,axis=0)
datay_echocardiogram=np.delete(datay_echocardiogram,flag)
#print(data,label)
imputer = SimpleImputer(missing_values = -1, strategy = "mean")
Fit = imputer.fit(dataX_echocardiogram)
dataX_echocardiogram = imputer.transform(dataX_echocardiogram)
#----------------------------------------------------------------------
cancer=datasets.load_breast_cancer()
dataX_breast=cancer.data
datay_breast=cancer.target

#-------------------------------------------------------------------

tmp = np.loadtxt("mammographic_masses.data", dtype=np.str, delimiter=",")
dataX_mammographic = tmp[:,0:-1].astype(np.int32)#加载数据部分S
datay_mammographic = tmp[:,-1].astype(np.int32)#加载类别标签部分

imputer = SimpleImputer(missing_values = -1, strategy = "mean")
Fit = imputer.fit(dataX_mammographic)
dataX_mammographic = imputer.transform(dataX_mammographic)
dataX_mammographic=dataX_mammographic.astype(int)

#--------------------------------------------------------------------

tmp = np.loadtxt("adult.test", dtype=np.str, delimiter=",")
dataX_adult = tmp[:,0:-1].astype(np.int32)#加载数据部分S
datay_adult = tmp[:,-1].astype(np.int32)#加载类别标签部分

#print(data,label)
imputer = SimpleImputer(missing_values = -1, strategy = "mean")
Fit = imputer.fit(dataX_adult)
dataX_adult = imputer.transform(dataX_adult)
dataX_adult=dataX_adult.astype(int)
data_index=np.array(range(len(dataX_adult)),dtype='int32')
np.random.shuffle(data_index)
dataX_adult=dataX_adult[data_index][:1000]
datay_adult=datay_adult[data_index][:1000]

datasets=[[dataX_credit,datay_credit],[dataX_echocardiogram,datay_echocardiogram],
        [dataX_breast,datay_breast],[dataX_mammographic,datay_mammographic],[dataX_adult,datay_adult]]



# dataX=np.array([],dtype='float')
# datay=np.array([],dtype='int32')



# i=0
# j=0
# ones=0
# zeros=0
# while len(data_index)<8000:
#     if label[i]==0 and zeros<4000:
#         data_index=np.append(data_index,i)
#         zeros+=1
#     if label[i]==1 and ones<4000:
#         data_index=np.append(data_index,i)
#         ones+=1
#     i+=1

# dataX=data[data_index]
# datay=label[data_index]


