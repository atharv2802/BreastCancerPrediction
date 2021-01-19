# -*- coding: utf-8 -*-

#importing essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import pickle

#importing dataset
df = pd.read_csv('breast_cancer_dataset.csv')

"""## **Preparing Data** & **EDA**"""

print('Data Shape',df.shape)

df.info()

df.head()

df.isnull().sum()

"""As in the column Unnamed:32, there are 569 missing entries so we will drop the column unnamed:32.

Also replacing M with 1 and B with 0.
"""

df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
df.head()

df['diagnosis'] = df['diagnosis'].replace({'M':1,'B':0})
#Change Diagnosis to be last column
df['target'] = df['diagnosis'].copy()
df.drop(['diagnosis'],axis = 1,inplace = True)
df.head()

df.isnull().sum()

"""**No null value present in the dataset**"""

Y = df.target
X = df.drop('target', axis=1)
X.shape

Y

#VISUALIZE NUMBER OF BENIGN AND MALIGNANT CASES

sns.countplot(df['target'], label='Count')

B, M = df['target'].value_counts()
print('Benign: ',B)
print('Malignant : ',M)

corr = df.corr()

plt.figure(figsize =(20,32))
n = 0
for i in list(df.iloc[:,:-1].columns):
        n += 1
        plt.subplot(11,3,n)
        plt.subplots_adjust(hspace = 0.5,wspace = 0.2)
        sns.boxplot(x=df[i])
plt.show()

corr_target = (df[df.columns[0:]].corr()['target'][:-1]).to_frame()
plt.figure(1,figsize =(20,12))
sns.barplot(y = corr_target.index,x = corr_target['target'],data = corr_target, orient = "h")
plt.title('Correlation of Target to other features in mean')
plt.ylabel('Correlation with Target')
plt.xlabel('Features')
plt.show()

plt.figure(figsize = (12,12))
sns.heatmap(df.corr(),annot=True,cmap = 'coolwarm');
plt.title("Correlation Between Features");
plt.show();



"""# **MODEL BUILDING**

MODELS USED:
1.   Random Forest
2. Decision Tree
3. Logistic Regression
4. SVM
5. K Nearest neighbour
"""

#importing necessary model libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import itertools

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

#import plotly.graph_objects as go

#test-train split

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = tts(X,y,test_size = 0.2,random_state = 7)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

"""## Random Forest Classifier"""

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
yhat_rf = clf_rf.predict(X_test)
yhat_proba_rf = clf_rf.predict_proba(X_test)

ac_rf = accuracy_score(y_test, yhat_rf)
print('Accuracy Score: ', ac_rf)

cm_rf = confusion_matrix(y_test, yhat_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='coolwarm', xticklabels='BM', yticklabels='BM')
plt.xlabel('Predicted Values - Y Hat')
plt.ylabel('Actual Values - Y')
plt.title('Confusion Matrix - Random Forest')

f1_rf = f1_score(y_test, yhat_rf, average='weighted') 
print('F1 Score: ', f1_rf)

"""## Decision Tree"""

clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
clf_dt.fit(X_train, y_train)
yhat_dt = clf_dt.predict(X_test)
yhat_proba_dt = clf_dt.predict_proba(X_test)

metrics.accuracy_score(yhat_dt, y_test)
ac_dt = metrics.accuracy_score(yhat_dt, y_test)
print("DecisionTrees's Accuracy: ", ac_dt)

cm_dt = confusion_matrix(y_test, yhat_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='coolwarm', xticklabels='BM', yticklabels='BM')
plt.xlabel('Predicted Values - Y Hat')
plt.ylabel('Actual Values - Y')
plt.title('Confusion Matrix - Decision Tree')

f1_dt = f1_score(y_test, yhat_dt, average='weighted') 
print('F1 Score: ', f1_dt)

"""## Logistic Regression"""

clf_lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
clf_lr

clf_lr.fit(X_train, y_train)
yhat_lr = clf_lr.predict(X_test)
yhat_proba_lr = clf_lr.predict_proba(X_test)

ac_lr = accuracy_score(y_test, yhat_lr)
print('Accuracy Score: ', ac_lr)

cm_lr = confusion_matrix(y_test, yhat_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='coolwarm', xticklabels='BM', yticklabels='BM')
plt.xlabel('Predicted Values - Y Hat')
plt.ylabel('Actual Values - Y')
plt.title('Confusion Matrix - Logistic Regression')

f1_lr = f1_score(y_test, yhat_lr, average='weighted') 
print('F1 Score: ', f1_lr)

print (classification_report(y_test, yhat_lr))

"""## SVM"""

clf_svm = svm.SVC(kernel='rbf', probability=True)
clf_svm.fit(X_train, y_train)
yhat_svm = clf_svm.predict(X_test)
yhat_proba_svm = clf_svm.predict_proba(X_test)

ac_svm = accuracy_score(y_test, yhat_svm)
print('Accuracy Score: ', ac_svm)

cm_svm = confusion_matrix(y_test, yhat_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='coolwarm', xticklabels='BM', yticklabels='BM')
plt.xlabel('Predicted Values - Y Hat')
plt.ylabel('Actual Values - Y')
plt.title('Confusion Matrix - Support Vector Machine')

f1_svm = f1_score(y_test, yhat_svm, average='weighted')
print('F1 Score: ', f1_svm)

"""## K - nearest neighbour"""

k = 7
clf_knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat_knn = clf_knn.predict(X_test)
yhat_proba_knn = clf_knn.predict_proba(X_test)

ac_knn = metrics.accuracy_score(y_test, yhat_knn)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, clf_knn.predict(X_train)))
print("Test set Accuracy: ", ac_knn)

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
# ConfusionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    clf_knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat_knn=clf_knn.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat_knn)

    
    std_acc[n-1]=np.std(yhat_knn==y_test)/np.sqrt(yhat_knn.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1) 

cm_knn = confusion_matrix(y_test, yhat_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='coolwarm', xticklabels='BM', yticklabels='BM')
plt.xlabel('Predicted Values - Y Hat')
plt.ylabel('Actual Values - Y')
plt.title('Confusion Matrix - K Nearest Neighbors')

f1_knn = f1_score(y_test, yhat_knn, average='weighted') 
print('F1 Score: ', f1_knn)

"""# **Comparing Results**"""

models_n = {'Algorithm' : ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Support Vector Machine', 'K Nearest Neighbor'],
     'Accuracy_Score' : [ac_rf, ac_dt, ac_lr, ac_svm, ac_knn],
    'F1_Score' : [f1_rf, f1_dt, f1_lr, f1_svm, f1_knn]}
df_accuracy = pd.DataFrame(data=models_n)
df_accuracy


#saving model to the user system
pickle.dump(clf_rf, open('bc_model.pkl','wb'))
#loading the model
bc_model = pickle.load(open('bc_model.pkl','rb'))

# checking the working of model onexample values.
print(bc_model.predict([[17.99,	10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095,	0.9053,	8.589,	153.4,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	0.006193,	25.38,	17.33,	184.6,	2019,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189,
]]))
