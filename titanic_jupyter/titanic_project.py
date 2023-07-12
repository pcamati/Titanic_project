import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

full_train_data = pd.read_csv('train.csv')

full_train_data.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
full_train_data['Sex'] = full_train_data['Sex'].apply(lambda x: 1 if x=='female' else -1)
full_train_data['Embarked'] = full_train_data['Embarked'].apply(lambda x: 1 if x=='C' else 0 if x=='Q' else -1)

condition = full_train_data['Age'].apply(lambda x: np.isnan(x))
full_train_data = full_train_data.drop(full_train_data[condition].index)

X = full_train_data.drop(columns=['Survived'])
y = full_train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
prediction_rfc = rfc.predict(X_test)

print(classification_report(y_test,prediction_rfc))
print(confusion_matrix(y_test,prediction_rfc))

clf = svm.SVC()
clf.fit(X_train,y_train)
prediction_clf = clf.predict(X_test)

print(classification_report(y_test,prediction_clf))
print(confusion_matrix(y_test,prediction_clf))

score_rfc = accuracy_score(y_test,prediction_rfc)
score_clf = accuracy_score(y_test,prediction_clf)

print(score_rfc)
print(score_clf)