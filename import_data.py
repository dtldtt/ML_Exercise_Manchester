import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

tmp = np.loadtxt("default of credit card clients.csv", dtype=np.str, delimiter=",")
data = tmp[2:,1:-1].astype(np.float)#加载数据部分
label = tmp[2:,-1].astype(np.int32)#加载类别标签部分
#print(data,label)

imputer = SimpleImputer(missing_values = 0.0000e+00, strategy = "mean")
Fit = imputer.fit(data)
new_data = imputer.transform(data)
#print(new_data)

dataX=new_data
datay=label