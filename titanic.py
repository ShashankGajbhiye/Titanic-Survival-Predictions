# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:24:37 2019

@author: Shashank
"""

"""-----PART 1-----"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the Datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test1 = pd.read_csv('test.csv')
train.head()
test.head()

# Setting the 'PassengerId' as Index in Train and Test Dataset
train.set_index(['PassengerId'], inplace = True)
test.set_index(['PassengerId'], inplace = True)

# Checking for Null Values in the Dataset
train.isnull().sum()
test.isnull().sum()
train.dtypes
#Visualising the Missing Values
import missingno as mn
mn.matrix(train)
mn.matrix(test)

"""-----PART 2-----"""

# Taking Care of the Missing Data
# Imputing the Column 'Age'
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
#For Dataset 'Train'
missingvalues = missingvalues.fit(train.iloc[:, [4]])
train.iloc[:, [4]] = missingvalues.transform(train.iloc[:, [4]])
#For Dataset 'Test'
missingvalues = missingvalues.fit(test.iloc[:, [3]])
test.iloc[:, [3]] = missingvalues.transform(test.iloc[:, [3]])

# Imputing the Column 'Embarked' in Train
train.Embarked.value_counts()
train.Embarked.fillna('S', inplace = True)

# Imputing the Column 'Fare' in Test
test.Fare.fillna(test.Fare.mean(), inplace = True)

# Dropping 'Cabin' Column since it contains 80% Missing Values
train.drop(['Cabin'], axis = 1, inplace = True)
test.drop(['Cabin'], axis = 1, inplace = True)
#Checking for Missing Data
train.isnull().sum()
test.isnull().sum()

"""-----PART 3-----"""

# Encoding the Categorical Values
train['Sex'] = train.Sex.apply(lambda x:0 if x == 'female' else 1)
test['Sex'] = test.Sex.apply(lambda x:0 if x == 'female' else 1)
train.Sex.head()
test.Sex.head()

# Removing Outliers from 'Fare'
sns.boxplot('Survived', 'Fare', data = train)
train['Fare'] = train[train['Fare'] <= 400]
test['Fare'] = test[test['Fare'] <= 400]

"""-----PART 4-----"""

# Feature Engineering / Feature Scaling
train['family_size'] = train['SibSp'] + train['Parch'] + 1
test['family_size'] = test['SibSp'] + test['Parch'] + 1

# Creating Categories according to family_size
def family_group(size):
    a = ''
    if(size <= 1):
        a = 'alone'
    elif(size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
train['family_group'] = train.family_size.map(family_group)
test['family_group'] = test.family_size.map(family_group)
train.head()
train.family_group.value_counts()
test.family_group.value_counts()

# Creating Categories according to Age
def age_group(age):
    a = ''
    if(age <= 1):
        a = 'infant'
    elif(age <= 4):
        a = 'small'
    elif(age <= 14):
        a = 'child'
    elif(age <= 25):
        a = 'young'
    elif(age <= 40):
        a = 'adult'
    elif(age <= 55):
        a = 'mid-age'
    else:
        a = 'old'
    return a
train['age_group'] = train.Age.map(age_group)
test['age_group'] = test.Age.map(age_group)
train.age_group.value_counts()
test.age_group.value_counts()

# Creating Categories according to Fare per person
train['fare_per_person'] = train['Fare']/train['family_size']
test['fare_per_person'] = test['Fare']/test['family_size']

def fare_group(fare):
    a = ''
    if(fare <= 10):
        a = 'low'
    elif(fare <= 20):
        a = 'mid'
    elif(fare <= 45):
        a = 'high'
    else:
        a = 'very-high'
    return a
train['fare_group'] = train.fare_per_person.map(fare_group)
test['fare_group'] = test.fare_per_person.map(fare_group)
train.fare_group.value_counts()
test.fare_group.value_counts()

# Creating the Dummy Variables
train = pd.get_dummies(train, columns = ['Embarked', 'family_group', 'age_group', 'fare_group'], drop_first = True)
test = pd.get_dummies(test, columns = ['Embarked', 'family_group', 'age_group', 'fare_group'], drop_first = True)

# Dropping unnecessary Columns
train.drop(['Name', 'Ticket', 'Fare', 'fare_per_person', 'family_size'], axis = 1, inplace = True)
test.drop(['Name', 'Ticket', 'Fare', 'fare_per_person', 'family_size'], axis = 1, inplace = True)

"""-----Part 5-----"""

x = train.drop('Survived', 1)
y = train['Survived']

# Applying the XGBoost Classifier
from xgboost import XGBClassifier
xgb = XGBClassifier()
from sklearn.model_selection import cross_val_score
score = cross_val_score(xgb, x, y, n_jobs = 1, scoring = 'accuracy', cv = 10)
print(score)
round(np.mean(score)*100, 2)
xgb.fit(x, y)

# Predicting the Test Set Results
y_pred = xgb.predict(test)
print(y_pred)
print(len(y_pred))

# Creating the Submission DataFrame
submission = pd.DataFrame({
        'PassengerId' : test1['PassengerId'],
        'Survived' : y_pred})
    
# Checking the Survived Passengers according to Sex
test['Predictions'] = y_pred
test[(test['Predictions'] == 0)]['Sex'].value_counts().plot(kind = 'bar')