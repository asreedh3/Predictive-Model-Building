# -*- coding: utf-8 -*-
"""

@author: Ashlin
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

#Creation of a user defined Confusion Matrix Function
def confusionmatrix(Actual,Predicted):
    cmatrix=pd.DataFrame({'Actual':Actual,'Predicted':Predicted})
    cross_tab=pd.crosstab(cmatrix['Actual'],cmatrix['Predicted'])
    print cross_tab



mydata=pd.read_csv('Train.csv')
X=mydata.iloc[:,1:67]
y=mydata.iloc[:,-1]
labels=X.columns.values

#Sampling Procedures
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

#feature selection model
rf=RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=2000)
rf.fit(X,y)
feature_selection_model=SelectFromModel(rf,prefit=True)
X = feature_selection_model.transform(X)
X_test = feature_selection_model.transform(X_test)

for feature_list_index in feature_selection_model.get_support(indices=True):
    print(labels[feature_list_index])


# Randomized Grid Search
seed=2000
n_estimators =[int(x) for x in np.linspace(start=40,stop=200, num=15)]
max_features=['sqrt','auto','log2']
max_depth=[int(x) for x in np.linspace(start=1,stop=10,num=6)]


random_parameters={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'random_state':[seed]}
modelnow=RandomForestClassifier()
rf_random=RandomizedSearchCV(modelnow,random_parameters,n_iter=200,cv=10,n_jobs=4,verbose=2)
rf_random.fit(X,y)
rf_bestrandom=rf_random.best_estimator_
yhat=rf_bestrandom.predict(X_test)
print("Best Parameters"),rf_random.best_params_
print("Training Score"),rf_bestrandom.score(X,y)
print("Testing Score"),rf_bestrandom.score(X_test,y_test)
confusionmatrix(y_test,yhat)

# Grid Search

seed=2000
parameters={'n_estimators': [130,150,165,180],'random_state':[seed],'oob_score':[True],'max_depth':[2,4,6,8,10],'max_features':['sqrt','log2','auto']}
modelnow=RandomForestClassifier()
rf_grid=GridSearchCV(modelnow,parameters,cv=10,n_jobs =4,verbose=2)
rf_grid.fit(X,y)
rf_best=rf_grid.best_estimator_
yhat=rf_best.predict(X_test)
print("Best Parameters"),rf_grid.best_params_
print("Training Score"),rf_best.score(X,y)
print("Testing Score"),rf_best.score(X_test,y_test)

confusionmatrix(y_test,yhat)

#ROC and Performance Metrics
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
print matthews_corrcoef(y_test,yhat)