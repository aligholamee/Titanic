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

# Drop the PassengerId column
train_data.drop('PassengerId', axis=1, inplace=True)

# Get the shape of data
separate_output("Train/Test Shapes -- Dropped PassengerId")
print(train_data.shape)
print(test_data.shape)

# General analysis of data
separate_output("General Data Knowledge -- Dropped PassengerId")
print(train_data.describe())

# Check if the gender affect the survivals
# Plot the figures for male and female
fig = plt.figure(figsize=(8, 4))
fig.add_subplot(121)
train_data.Survived[train_data['Sex'] == 'male'].value_counts().plot(kind='pie')
fig.add_subplot(122)
train_data.Survived[train_data['Sex'] == 'female'].value_counts().plot(kind='pie')
plt.show()