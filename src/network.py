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

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "./data/"

TRAIN_DATASET = "train.csv"
TEST_DATASET = "test.csv"

# Load the train data
TRAIN_DATASET = pd.read_csv(DATA_PATH + TRAIN_DATASET)

# Load the test data
TEST_DATASET = pd.read_csv(DATA_PATH + TEST_DATASET)

# Create the model
DT_MODEL = DecisionTreeClassifier()

# Grab the targets
TARGETS = TRAIN_DATASET["Survived"]
TRAIN_DATASET.drop('Survived')

# Train the model
DT_MODEL.fit(TRAIN_DATASET, TARGETS)

# Prediction
print(DT_MODEL.predict(TEST_DATASET))
# # Check if there is any NaN values in the dataset
# print(TRAIN_DATASET.isnull().values.any())

# # Fill the missing data using filna :D
# TRAIN_DATASET["Cabin"].fillna("Missing")
# TRAIN_DATASET.fillna(0)

# # Print the whole dataset
# print(TRAIN_DATASET.loc[(TRAIN_DATASET["Sex"] == "male") & (TRAIN_DATASET["Pclass"] < 2)])

# Describe the data frame
# print(TRAIN_DATASET.describe())
# print(TRAIN_DATASET.loc[(TRAIN_DATASET["Sex"]=="male") & (TRAIN_DATASET["Pclass"] < 2)])
# Boolean Indexing 
#TRAIN_DATASET.loc([])