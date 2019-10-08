import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
import numpy as np

# import data and create train dataset and test dataset
iris=datasets.load_iris()
dataX=iris.data
datay=iris.target
trainX=np.concatenate((dataX[:30],dataX[50:80],dataX[100:130]),axis=0)
trainy=np.concatenate((datay[:30],datay[50:80],datay[100:130]),axis=0)
testX=np.concatenate((dataX[30:50],dataX[80:100],dataX[130:150]),axis=0)
testy=np.concatenate((datay[30:50],datay[80:100],datay[130:150]),axis=0)

#epochs from 10 to 140, plus 10 each loop
epochs=list(range(10,210,10))
#train errors and test errors for each epoch
train_scores=[]
test_scores=[]

for e in epochs:
    clf=SGDClassifier(loss="log",learning_rate="constant",eta0=0.001,max_iter=e)
    clf.fit(trainX,trainy)
    test_scores.append(clf.score(testX,testy))
    train_scores.append(clf.score(trainX,trainy))

plt.plot(epochs,test_scores,linewidth=3,c='red')
plt.plot(epochs,train_scores,linewidth=3,c='green')
plt.axis([0,epochs[len(epochs)-1],0,1])
plt.show()
    
