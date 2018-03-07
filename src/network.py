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

# coding: utf-8

# Import Pandas and Numpy libraries. These are the main libraries for array computations.

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# Define the data root to be take train data from.

# In[2]:


DATA_ROOT = './data/'


# Load the train and test data using read_csv function in the Pandas library.

# In[3]:


train_data = pd.read_csv(DATA_ROOT + 'train.csv')
test_data = pd.read_csv(DATA_ROOT + 'test.csv')


# A function to separate the different outputs of each section. This will be used to display the data in the terminal in a readable way.

# In[4]:


def separate_output(str):
    '''
        Displays an string as an argument in a clear form
    '''
    SHARP_COUNT = 100
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


# Find out how the data looks like. This helps us to get an intuition of features inside the datasets. This is done using the shape class member of a Pandas dataframe.

# In[5]:


separate_output("Train/Test Shapes")
print(train_data.shape)
print(test_data.shape)


# This will provide some statistical knowledge about the data. We can observe the mean, variance, max and minimum of data for each feature. This can be used for data normalization and preprocessing. We have used the describe method from the Pandas dataframe class.

# In[6]:


separate_output("General Data Knowledge")
train_data.describe()


# Some features like PassengerId, Name, Ticket, Cavbin and Embarked can be removed from the dataset.

# In[7]:


# These columns will be dropped
DROPPED_COLS = ['PassengerId',
                'Ticket',
                'Cabin',
                'Embarked']

# Drop the PassengerId column
train_data.drop(DROPPED_COLS, axis=1, inplace=True)


# In[8]:


# Get the shape of data
separate_output("Train/Test Shapes -- Dropped 5 Columns")
print(train_data.shape)
print(test_data.shape)


# Let's plot the pie chart of the sex. We want to analyze if the gender affects the survivals or not. At the first glance, it seems that gender could be a good feature for prediction. These plots confirm this idea.

# In[9]:


import matplotlib.pyplot as plt

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
plt.show()


# Let's check the datatypes to make sure there are no more objects left in the dataset. Objects are representing a text most of the times(Categorical Data). We obtain this using select_dtypes from the Pandas Dataframe class.

# In[10]:


# Let's see if there are any more categorical data left
separate_output("Datatypes")
print(train_data.select_dtypes(include=[object]))


# We use LabelEncoder to convert the categorical data into the numerical form. To do this, simply create an object of the LabelEncoder class and call the fit_transform function on the desired data column in the dataset.

# In[11]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

# Convert the categorical data into numerical form
train_data['Sex'] = LabelEncoder().fit_transform(train_data['Sex'])


# Split the titles from the passenger names which is itself a feature but also help in calculating missing median age values.

# In[12]:


train_data['Name'] = train_data['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train_data['Name'].unique()
titles


# Sneaking into the Age column, we can see there are some NaN numbers. These are called missing values. In order to increase the number of data samples, we need to fill these NaN values with an appropriate values. Fill the NaN values of Age using median values related to its title.

# In[13]:


train_data['Age'].fillna(-1, inplace=True)

medians = dict()
for title in titles:
    median = train_data.Age[(train_data["Age"] != -1) & (train_data['Name'] == title)].median()
    medians[title] = median
    
for index, row in train_data.iterrows():
    if row['Age'] == -1:
        train_data.loc[index, 'Age'] = medians[row['Name']]
train_data.head()


# Before transforming the Name column into the numerical form, we'll be excavating the distribution of our training data with respect to the Names. We will assign the numbers to these Names according to the distribution of each of these titles shown below.

# In[14]:


fig = plt.figure(figsize=(15,6))

i=1
for title in train_data['Name'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Title : {}'.format(title))
    train_data.Survived[train_data['Name'] == title].value_counts().plot(kind='pie')
    i += 1


# In[15]:


REPLACEMENTS = {
    'Don': 0,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 7,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
}

train_data['Name'] = train_data['Name'].apply(lambda x: REPLACEMENTS.get(x))


# We can also fill the NaN values of Fare using by its correlation with Ticket Class.

# In[16]:


train_data['Fare'].fillna(-1, inplace=True)
medians = dict()
for pclass in train_data['Pclass'].unique():
    median = train_data.Fare[(train_data["Fare"] != -1) & (train_data['Pclass'] == pclass)].median()
    medians[pclass] = median
for index, row in train_data.iterrows():
    if row['Fare'] == -1:
        train_data.loc[index, 'Fare'] = medians[row['Pclass']]


# Plot the distribution of our data with respect to each class of tickets.

# In[17]:


fig = plt.figure(figsize=(15,4))

i=1
for pclass in train_data['Pclass'].unique():
    fig.add_subplot(1, 3, i)
    plt.title('Class : {}'.format(pclass))
    train_data.Survived[train_data['Pclass'] == pclass].value_counts().plot(kind='pie')
    i += 1


# The classes are numeric already. Let's analyze the next feature.

# In[18]:


fig = plt.figure(figsize=(15,8))
i = 0
for parch in train_data['Parch'].unique():
    fig.add_subplot(2, 4, i+1)
    plt.title('Parents / Child : {}'.format(parch))
    train_data.Survived[train_data['Parch'] == parch].value_counts().plot(kind='pie')
    i += 1


# In[19]:


CP_REPLACEMENTS = {
    6: 0,
    4: 0,
    5: 1,
    0: 2,
    2: 3,
    1: 4,
    3: 5
}
train_data['Parch'] = train_data['Parch'].apply(lambda x: CP_REPLACEMENTS.get(x))


# In[20]:


train_data.head()


# Now the data is almost ready to be trained. We can start training using the predefined models in sklearn library. Following models are used in this example.

# In[21]:


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# An array containing models
MODELS = [
    MLPClassifier(),
    AdaBoostClassifier(),
    SVC(),
    QuadraticDiscriminantAnalysis(),
    GaussianProcessClassifier()
]


# Since the labels for the test_data is not available, we use train_data for both training and testing. We can use the function train_test_split to split 20% of data for test and 80% for training. The actual labels for the training set is first extracted. The Survived column is dropped. Finally, the train_test_split is called on the tranining data with respective labels.

# In[22]:


from sklearn.model_selection import train_test_split

# Split the train and test data
train_labels = train_data['Survived']
train_data.drop('Survived', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)


# We have used the recall score and the f1 score to represent the performance of each of these classifiers. The fit functon can be called from each classifier object. It is used to train some data on that specific classifier. The predict function returns the predicted labels.

# In[23]:


from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

for model in MODELS:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    rscore = recall_score(y_test, prediction)
    f1score = f1_score(y_test, prediction)
    score = model.score(X_test, y_test)
    print(score)
    print("Recall: ", rscore)
    print("F-1 Score: ", f1score)

