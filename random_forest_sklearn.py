from sklearn.ensemble import RandomForestClassifier
from import_data import dataX,datay

trainX=dataX[:1000]
trainy=datay[:1000]
testX=dataX[1000:2000]
testy=datay[1000:2000]
test_score=[]
num=[1,3,5,7,9,11,13]


    


def score(testX,testy,clf):
    i=0
    right=0
    while i<len(testX):
        temp=clf.predict(testX)
        if temp[0]==testy[i]:
            right+=1
        i+=1
    return (right/len(testX))


for n in num:
    clf=RandomForestClassifier(max_depth=6,n_estimators=n)
    clf.fit(trainX,trainy)
    test_score.append(score(testX,testy,clf))

print(test_score)