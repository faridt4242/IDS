import pandas as pd  
import numpy as np  
import statsmodels.api as sm
import sklearn.metrics 
import seaborn as sbs
import matplotlib.pyplot as plt  
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from pandas import DataFrame
from sklearn.model_selection import cross_val_score

#Classifiers 
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial') 
NB = MultinomialNB()
KNN = KNeighborsClassifier()
DT = DecisionTreeClassifier()
ADA = AdaBoostClassifier()
RF = RandomForestClassifier(n_estimators=100) 

# Importing datasets
trainset = pd.read_csv('data/train.csv')
testset = pd.read_csv('data/test.csv')

# Separating label from variables for training set
xtrain = trainset.iloc[:,0:42]
ytrain = trainset.iloc[:,43]

# Separating label from variables for testing set
xtest = testset.iloc[:,0:42]
ytest = testset.iloc[:,43]

# Removing highly correlated features which results in 30 features
corr = xtrain.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

columnsToUse = xtrain.columns[columns] 
xtrain = xtrain[columnsToUse]
xtest = xtest[columnsToUse]

# LR
LR.fit(xtrain, ytrain)
tet = LR.predict(xtest)
print("LR performance:")
print("accuracy:")
print(accuracy_score(ytest, tet))
print("precision:")
print(sklearn.metrics.precision_score(ytest, tet))
print("recall:")
print(sklearn.metrics.recall_score(ytest, tet))
print("f1-score:")
print(sklearn.metrics.f1_score(ytest, tet))
print("\n")


# NB
NB.fit(xtrain, ytrain)
tet = NB.predict(xtest)
print("NB performance:")
print("accuracy:")
print(accuracy_score(ytest, tet))
print("precision:")
print(sklearn.metrics.precision_score(ytest, tet))
print("recall:")
print(sklearn.metrics.recall_score(ytest, tet))
print("f1-score:")
print(sklearn.metrics.f1_score(ytest, tet))
print("\n")


# KNN
KNN.fit(xtrain, ytrain)
tet = KNN.predict(xtest)
print("KNN performance:")
print("accuracy:")
print(accuracy_score(ytest, tet))
print("precision:")
print(sklearn.metrics.precision_score(ytest, tet))
print("recall:")
print(sklearn.metrics.recall_score(ytest, tet))
print("f1-score:")
print(sklearn.metrics.f1_score(ytest, tet))
print("\n")


# DT
DT.fit(xtrain, ytrain)
tet = DT.predict(xtest)
print("DT performance:")
print("accuracy:")
print(accuracy_score(ytest, tet))
print("precision:")
print(sklearn.metrics.precision_score(ytest, tet))
print("recall:")
print(sklearn.metrics.recall_score(ytest, tet))
print("f1-score:")
print(sklearn.metrics.f1_score(ytest, tet))
print("\n")


#ADAboost
ADA.fit(xtrain, ytrain)
tet = ADA.predict(xtest)
print("ADA performance:")
print("accuracy:")
print(accuracy_score(ytest, tet))
print("precision:")
print(sklearn.metrics.precision_score(ytest, tet))
print("recall:")
print(sklearn.metrics.recall_score(ytest, tet))
print("f1-score:")
print(sklearn.metrics.f1_score(ytest, tet))
print("\n")


# RF
RF.fit(xtrain, ytrain)
tet = RF.predict(xtest)
print("RF performance:")
print("accuracy:")
print(accuracy_score(ytest, tet))
print("precision:")
print(sklearn.metrics.precision_score(ytest, tet))
print("recall:")
print(sklearn.metrics.recall_score(ytest, tet))
print("f1-score:")
print(sklearn.metrics.f1_score(ytest, tet))
print("\n")
