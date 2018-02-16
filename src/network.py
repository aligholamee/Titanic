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

DATA_PATH = "./data/"

TRAIN_DATASET = "train.csv"
TEST_DATASET = "test.csv"

# Load the train data
TRAIN_DATASET = pd.read_csv(DATA_PATH + TRAIN_DATASET)

# Check if there is any NaN values in the dataset
print(TRAIN_DATASET.isnull().values.any())

# Insert a missing data into the cabin C148 and ticket 111369 in its Fare column
TRAIN_DATASET.loc[(TRAIN_DATASET["Cabin"] == "C148") & (TRAIN_DATASET["Ticket"] == "111369")] = None

# Fill the missing data using filna :D
TRAIN_DATASET.fillna(0)

# Print the whole dataset
print(TRAIN_DATASET.loc[(TRAIN_DATASET["Sex"] == "male") & (TRAIN_DATASET["Pclass"] < 2)])

# Describe the data frame
# print(TRAIN_DATASET.describe())
# print(TRAIN_DATASET.loc[(TRAIN_DATASET["Sex"]=="male") & (TRAIN_DATASET["Pclass"] < 2)])
# Boolean Indexing 
#TRAIN_DATASET.loc([])