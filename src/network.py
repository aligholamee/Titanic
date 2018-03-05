# ========================================
# [] File Name : network.py
#
# [] Creation Date : Februray 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Kaggle competition, predicting the survivals of the Titanic!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

DATA_ROOT = './data/'
SHARP_COUNT = 100

# Simple output separator for the terminal displays
def separate_output(str):
    '''
        Displays an string as an argument in a clear form
    '''
    print('\n')
    for i in range(SHARP_COUNT):
        if(i == SHARP_COUNT-1):
            print("#")
        else:
            print("#", end="")

    # Display info at the center
    for i in range(int((SHARP_COUNT/2-len(str)/2))):
        print("",end=" ")

    print(str)

    for i in range(SHARP_COUNT):
        print("#", end="")
    print('\n')

# This will fill the NaN values in columns
def fill_with_mean(df):
    return df.fillna(df.mean())

# Load the train and test data
train_data = pd.read_csv(DATA_ROOT + 'train.csv')
test_data = pd.read_csv(DATA_ROOT + 'test.csv')

# Get the shape of data
separate_output("Train/Test Shapes")
print(train_data.shape)
print(test_data.shape)

# General analysis of data
separate_output("General Data Knowledge")
print(train_data.describe())
print(train_data)

# These columns will be dropped
DROPPED_COLS = ['PassengerId',
                'Name',
                'Ticket',
                'Cabin',
                'Embarked']

# Drop the PassengerId column
train_data.drop(DROPPED_COLS, axis=1, inplace=True)

# Get the shape of data
separate_output("Train/Test Shapes -- Dropped 5 Columns")
print(train_data.shape)
print(test_data.shape)

# General analysis of data
separate_output("How Data Looks Like -- Dropped Some Columns")
print(train_data)

# Check if the gender affect the survivals
# Plot the figures for male and female
fig = plt.figure(figsize=(8, 4), dpi=120, facecolor='w', edgecolor='k')
fig.canvas.set_window_title("Analaysis of Gender Effect on Survivals")

male_survival = fig.add_subplot(121)
train_data.Survived[train_data['Sex'] == 'male'].value_counts().plot(kind='pie')
male_survival.set_title("Male Survivals")

female_survival = fig.add_subplot(122)
train_data.Survived[train_data['Sex'] == 'female'].value_counts().plot(kind='pie')
female_survival.set_title("Female Survivals")
# plt.show()

# Let's see the data types
separate_output("Datatypes")
print(train_data.select_dtypes(include=[object]))

# Convert the categorical data into numerical form
train_data['Sex'] = LabelEncoder().fit_transform(train_data['Sex'])

# Remove the NaN values from Age column
train_data = fill_with_mean(train_data)

# Scale the data to normalize the mean and variance
# ss = StandardScaler().fit(train_data)

# Display the data again
separate_output("Final Data")
columns_stack = [
    'Survived',
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]
# train_data = pd.DataFrame(ss.transform(train_data))
train_data.columns = columns_stack

print(train_data)

# Describe the status of final data
separate_output("Final Data Description")
print(train_data.describe())

# An array containing models
MODELS = [
    MLPClassifier(),
    AdaBoostClassifier(),
    SVC(),
    QuadraticDiscriminantAnalysis(),
    GaussianProcessClassifier()
]

# Split the train and test data
train_labels = train_data['Survived']
train_data.drop('Survived', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

for model in MODELS:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    rscore = recall_score(y_test, prediction)
    f1score = f1_score(y_test, prediction)

    print("Recall: ", rscore)
    print("F-1 Score: ", f1score)
    # score = recall_score(y_train, y_pred)
    # print(score)
