import numpy as np 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target
np.random.rand(0)
indices=np.random.permutation(len(iris_X))
# len(iris_X)=150
trainX=iris_X[indices[:80]]
trainy=iris_y[indices[:80]]
testX=iris_X[indices[80:]]
testy=iris_y[indices[80:]]

k_sets=list(range(31))[1:]
test_scores=[]

for k in k_sets:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainX,trainy)
    test_scores.append(knn.score(testX,testy))

plt.plot(k_sets,test_scores,linewidth=3,c='red')
plt.axis([0,k_sets[len(k_sets)-1],0.8,1])
plt.show()