import numpy as np

def model_function(X,stump):
    if X>stump:
        return 1
    else:
        return 0

def cal_error_rate(trainX,trainy,stump):
    error_num=0
    for X,y in zip(trainX,trainy):
        if model_function(X,stump)!=y:
            error_num=error_num+1
    return error_num/len(trainy)

def decision_stump(trainX,trainy,step_size=1,min_error=99999):
    max_X=np.max(trainX)
    min_X=np.min(trainX)
    x=min_X
    result=x
    while x<max_X:
        error_rate=cal_error_rate(trainX,trainy,x)
        if error_rate<min_error:
            min_error=error_rate
            result=x
        x=x+step_size
    return result


# exampleX=[98.79,93.64,42.89,87.91,97.9,47.63,92.72,60,58]
# exampley=[1,1,0,1,1,0,1,0,0]
# print(decision_stump(exampleX,exampley))