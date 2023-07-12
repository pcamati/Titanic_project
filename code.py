import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#setting up the raw dataframes
full_train_data = pd.read_csv('train.csv')
full_test_data = pd.read_csv('test.csv')

#Separate features from label
X = full_train_data.drop(['Survived'], axis=1)
y = full_train_data['Survived']

#Separate the raw training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def drop_columns(df):
    '''
    Removes the features 'Name', 'Cabin', and 'Ticket'
    from the titanic data.
    '''
    if 'Name' in df:
        df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
    else:
        pass
    return df

#Separate categorical and numerical features for pipelines
cat_cols = ['Sex','Embarked']
num_cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

#Define pipelines
pipe_cat = Pipeline([
    ('impute_cat', SimpleImputer(strategy='most_frequent')),
    ('encode_cat', OrdinalEncoder())
])
pipe_num = Pipeline([
    ('impute_num',SimpleImputer(strategy='mean'))
])
pipe_normalize = Pipeline([
    ('normalize',StandardScaler())
])

#two column-wise pipelines (can I do with one?)
ct = ColumnTransformer([
    ('pipe_cat', pipe_cat, cat_cols),
    ('pipe_num', pipe_num, num_cols)
])

norm = ColumnTransformer([
    ('normalize', pipe_normalize, ['Age', 'Fare'])],
    remainder='passthrough')

#Automate all data transformations
def prepare_for_estimator(df):  
    aux = pd.DataFrame(ct.fit_transform(drop_columns(df)), columns=cat_cols+num_cols)
    return norm.fit_transform(aux)

#Prepare train and test data
X_train = prepare_for_estimator(X_train)
X_test = prepare_for_estimator(X_test)

#Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(X_train,y_train)
pred_random_forest = random_forest.predict(X_test)

#Support Vector Machine Classifier
support_vector = SVC()
support_vector.fit(X_train,y_train)
pred_support_vector = support_vector.predict(X_test)

#Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
pred_decision_tree = decision_tree.predict(X_test)

# #Logistic Regression Classifier
# logistic_regression = LogisticRegression()
# logistic_regression.fit(X_train,y_train)
# pred_logistic_regression = logistic_regression.predict(X_test)

#Kneighbors Classifier
kneighbors = KNeighborsClassifier()
kneighbors.fit(X_train,y_train)
pred_kneighbors = kneighbors.predict(X_test)

#Gradian Boost Classifier
gradboost = GradientBoostingClassifier()
gradboost.fit(X_train,y_train)
pred_gradboost = gradboost.predict(X_test)

#Score predictions
score_random_forest = accuracy_score(y_test,pred_decision_tree)
score_support_vector = accuracy_score(y_test,pred_random_forest)
score_decision_tree = accuracy_score(y_test,pred_decision_tree)
# score_logistic_regression = accuracy_score(y_test,pred_logistic_regression)
score_kneighbors = accuracy_score(y_test,pred_kneighbors)
score_gradboost = accuracy_score(y_test,pred_gradboost)


# print(score_random_forest)
# print(score_support_vector)
# print(score_decision_tree)
# # print(score_logistic_regression)
# print(score_kneighbors)
# print(score_gradboost)

cross_val = 3

X_cross = prepare_for_estimator(X)
scores_random_forest = cross_val_score(random_forest, X_cross, y, cv=cross_val)
scores_support_vector = cross_val_score(support_vector, X_cross, y, cv=cross_val)
scores_decision_tree = cross_val_score(decision_tree, X_cross, y, cv=cross_val)
scores_kneighbors = cross_val_score(kneighbors, X_cross, y, cv=cross_val)
scores_gradboost = cross_val_score(gradboost, X_cross, y, cv=cross_val)

print(str(scores_random_forest.mean())+'+/-'+str(scores_random_forest.std()))
print(str(scores_support_vector.mean())+'+/-'+str(scores_support_vector.std()))
print(str(scores_decision_tree.mean())+'+/-'+str(scores_decision_tree.std()))
print(str(scores_kneighbors.mean())+'+/-'+str(scores_kneighbors.std()))
print(str(scores_gradboost.mean())+'+/-'+str(scores_gradboost.std()))

#prepare the prediction file
#create a final_prediction.csv file with two columns: PassengerId and Survived 
# final_prediction = full_test_data['PassengerId']
X = prepare_for_estimator(X)
random_forest_final = RandomForestClassifier(n_estimators=50)
random_forest_final.fit(X,y)
test_data = prepare_for_estimator(full_test_data)
pred = random_forest_final.predict(test_data)
pred = pd.Series(pred, name='Survived')
final_prediction = pd.concat([full_test_data['PassengerId'],pred], axis=1)
final_prediction.to_csv('final_prediction.csv', index=False)


