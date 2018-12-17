# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:01:09 2018

@author: Ashlin
"""
import pandas as pd
import numpy as np
from pandas_ml import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
def confusionmatrix(Actual,Predicted):
    cmatrix=pd.DataFrame({'Actual':Actual,'Predicted':Predicted})
    cross_tab=pd.crosstab(cmatrix['Actual'],cmatrix['Predicted'])
    print cross_tab



mydata=pd.read_csv('Train.csv')
X=mydata.iloc[:,1:67]
y=mydata.iloc[:,-1]
class_counts=y.value_counts()
df=X.dtypes
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=5000)
df_new=pd.concat([X_train,y_train],axis=1)
class_count=df_new.y.value_counts()

df_dominant = df_new[df_new.y==-1]
df_underrepresented = df_new[df_new.y==1]

df_dominant_downsampled = resample(df_dominant, replace=False,n_samples=1000,random_state=5000)

df_underrepresented_upsampled = resample(df_underrepresented,replace=True,n_samples=1000,random_state=5000)
df_upsampled = pd.concat([df_dominant_downsampled, df_underrepresented_upsampled])
df_upsampled.y.value_counts()

X=df_upsampled.iloc[:,0:66]
y=df_upsampled.iloc[:,-1]
X = pd.get_dummies(X)

X_test=pd.get_dummies(X_test)
#random forest base model
seed=2000
parameters={'n_estimators': [100,500,1000], 'max_features': ['sqrt','log2','auto'],'random_state':[seed],'oob_score':[True]}
modelnow=RandomForestClassifier()
rf_grid=GridSearchCV(modelnow,parameters,cv=5,n_jobs =4)
rf_grid.fit(X,y)
rf_best=rf_grid.best_estimator_
print("Best Parameters"),rf_grid.best_params_
print("Training Score"),rf_best.score(X,y)
print("Testing Score"),rf_best.score(X_test,y_test)
yhat=rf_best.predict(X_test)
confusionmatrix(y_test,yhat)
fpr, tpr, thresholds = roc_curve(y_test,yhat)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print f1_score(y_test,yhat)
#SVM Base model
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X_test=scaler.transform(X_test)

svc = SVC(random_state=seed)
svc_grid = GridSearchCV(svc, {'gamma':[0.001,0.01,0.1,1], 'C':[0.01,0.1,1]},return_train_score=True, n_jobs=4,cv=5,verbose=2)
svc_grid.fit(X,y)
svc_best=svc_grid.best_estimator_ 

print 'Best parameters are:',svc_grid.best_params_
print("The test accuracy is "),svc_best.score(X_test,y_test)
print("The training accuracy is"), svc_best.score(X,y)
yhat=svc_best.predict(X_test)
confusionmatrix(y_test,yhat)

#AdaBoost Base Model

parameters={'n_estimators':[50,100,500,1000],'random_state':[seed]}
ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
ada_grid=GridSearchCV(ada,parameters,n_jobs=4,cv=5)
ada_grid.fit(X,y)
ada_best=ada_grid.best_estimator_
print("Best Parameters"),ada_grid.best_params_
print("Training Score"),ada_best.score(X,y)
print("Testing Score"),ada_best.score(X_test,y_test)
yhat=ada_best.predict(X_test)
confusionmatrix(y_test,yhat)