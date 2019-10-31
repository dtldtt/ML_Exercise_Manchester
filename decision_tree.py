import numpy as np 
from sklearn import datasets
from decision_stump import decision_stump
from sklearn import metrics
from tree import Tree
import warnings
import matplotlib.pyplot as plt
import copy

warnings.filterwarnings("ignore")



# example_num=[50,200,400]
# train_scores=[]
# test_scores=[]

# for n in example_num:
#     np.random.seed()
#     indices=np.random.permutation(len(dataX))
#     #train_num=n*2//3
#     train_num=n
#     # trainX=dataX[indices[:train_num]]
#     # trainy=datay[indices[:train_num]]
#     # testX=dataX[indices[train_num:]]
#     # testy=datay[indices[train_num:]]
#     trainX=dataX[:n]
#     trainy=datay[:n]
#     testX=dataX[n:]
#     testy=datay[n:]
#     flags=[1]*(trainX.shape[1])
#     tree=build_tree(trainX,trainy,flags,depth=8)
#     train_scores.append(score(trainX,trainy,tree))
#     test_scores.append(score(testX,testy,tree))
# print(train_scores,test_scores)

# plt.plot(example_num,train_scores,linewidth=3,c='red')
# plt.plot(example_num,test_scores,linewidth=3,c='green')
# plt.show()




def build_tree(trainX,trainy,flags,tree=Tree(),depth=10,RF=0,K=0,gr=0):
    if depth==0 or len(np.unique(trainy))==1:
        count0=list(trainy).count(0)
        count1=list(trainy).count(1)
        if count0>count1:
            return 0
        else:
            return 1
    
    feature_score=0
    best_feature=0
    feature_split_point=0

    #flags=[1]*(trainX.shape[1])
    if RF==1:
        features=set()
        print(flags)
        # print("depth:",depth)
        # print("Before 1 loop")
        if flags.count(1)>K:
            while len(features)!=K:
                random_n=np.random.randint(0,trainX.shape[1])
                #print(random_n)
                # if flags.count(1)<7:
                #     print(random_n)
                if (random_n in features) or flags[random_n]==0:
                    continue
                else:
                    features.add(random_n)
        else:
            for i in flags:
                if i==1:
                    features.add(i)
        #print(features)
        #print("after 1 loop and before 2 loop")
        print("K features",features)

        if gr==0:
            max_info_gain=0
            for i in features:
                temp=metrics.mutual_info_score(trainX.T[i],trainy)
                if temp>max_info_gain:
                    max_info_gain=temp
                    best_feature=i
        elif gr==1:
            feature_info_gain=[]
            for i in features:
                feature_info_gain.append(metrics.mutual_info_score(trainX.T[i],trainy))

            mean=np.mean(feature_info_gain)
            good_features=[1]*len(feature_info_gain)
            j=0
            while j<len(feature_info_gain):
                if feature_info_gain[j]<mean:
                    good_features[j]=0
                j+=1
            j=0
            best_gain_ratio=0
            # for i in features:
            #     temp=gain_ratio(metrics.mutual_info_score(trainX.T[i],trainy),intrinsic_value(trainX.T[i]))
            #     if temp>best_gain_ratio:
            #         best_gain_ratio=temp
            #         best_feature=i
            while j<len(good_features):
                if good_features[j]==1:
                    temp=gain_ratio(feature_info_gain[j],intrinsic_value(trainX.T[list(features)[j]]))
                    if temp>best_gain_ratio:
                        best_feature=list(features)[j]
                        best_gain_ratio=temp
                j+=1

        else:
            for i in features:
                # feature_divides=np.percentile(trainX.T[i],(25,50,75),interpolation='midpoint')
                # train_feature=np.array([],dtype='int32')
                # for eachX in trainX.T[i]:
                #     if eachX<feature_divides[0]:
                #         train_feature=np.append(train_feature,1)
                #     elif eachX<feature_divides[1]:
                #         train_feature=np.append(train_feature,2)
                #     elif eachX<feature_divides[2]:
                #         train_feature=np.append(train_feature,3)
                #     else:
                #         train_feature=np.append(train_feature,4)
                #print(len(trainX.T[i]),len(train_feature),len(trainy))
                current_feature=metrics.mutual_info_score(trainX.T[i],trainy)
                if current_feature>feature_score:
                    feature_score=current_feature
                    best_feature=i
                i=i+1
            
            feature_split_point=decision_stump(trainX.T[best_feature],trainy)
        if gr==0 or gr==1:
            feature_divides=[]
            split_score=0
            j=0
            temp=list(copy.deepcopy(trainX.T[best_feature]))
            temp.sort()
            #print("before 2.1 loop")
            while j<len(trainX.T[best_feature])-1:              
                feature_divides.append((temp[j]+temp[j+1])/2)
                j+=1
            #print("after 2.1 loop and before 2.2 loop")

            k=0
            train_feature=np.full(len(trainX.T[best_feature]),1,dtype=int)
            while k<len(feature_divides):
                
                train_feature[k]=0
                current_split=metrics.mutual_info_score(train_feature,trainy)
                if current_split>split_score:
                    feature_split_point=feature_divides[k]
                k+=1
        #print(len(trainX.T[i]),len(train_feature),len(trainy))
        #print("after 2.2 loop")
    

    #print(trainX.T[best_feature],trainy)
    flags[best_feature]=0
    #print(depth)
    #print(flags)   
    #threshold=decision_stump(trainX.T[best_feature],trainy,step_size=0.1)
    threshold=feature_split_point
    print("feature:",best_feature)
    print("split point",threshold)
    #print(threshold)
    tree.data=[best_feature,threshold]
    left_tree=Tree()
    right_tree=Tree()
    leftsub=np.array([],dtype='int32')
    rightsub=np.array([],dtype='int32')
    
    # split into 2 parts: leftsub and rightsub
    #print("before split loop")
    i=0
    while i<(trainX.shape[0]):
        if trainX[i,best_feature]<=threshold:
            leftsub=np.append(leftsub,i)
        else:
            rightsub=np.append(rightsub,i)
        i=i+1
    #print(leftsub,rightsub)
    if len(leftsub)>0:
        left_tree=build_tree(trainX[leftsub],trainy[leftsub],flags,tree=left_tree,depth=depth-1,RF=RF,K=K,gr=gr)
        tree.left=left_tree
    if len(rightsub)>0:
        right_tree=build_tree(trainX[rightsub],trainy[rightsub],flags,tree=right_tree,depth=depth-1,RF=RF,K=K,gr=gr)
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

def intrinsic_value(X):
    uniques=np.unique(X)
    data=list(X)
    result=0
    for u in uniques:
        result+=data.count(u)/len(X)+np.log2(data.count(u)/len(X))
    return -result

def gain_ratio(info_gain,intrin_val):
    return info_gain/intrin_val