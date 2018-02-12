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

print(TRAIN_DATASET.loc[(TRAIN_DATASET["Sex"]=="male") & (TRAIN_DATASET["Pclass"] < 2)])
# Boolean Indexing 
#TRAIN_DATASET.loc([])