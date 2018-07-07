

# K-NEAREST
#/////////////////////////////// imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read dataset to pandas dataframe
dataset = pd.read_csv("data.csv")
dataset.head()

#How to pre-process the data?
# fix dataset
# 0,4,9,10,11 is continous so select only descrete columns
cat_columns = []
for i in [1,2,4,5,6,7,8,12,13]:
    cat_columns.append(dataset.columns[i])

# replace empty values
dataset = dataset.replace(" ?", pd.NaT)
dataset.dropna(inplace=True)


for i in cat_columns:
    dataset[i] = dataset[i].astype('category')

 # represet them using numerical values
for i in cat_columns:
    dataset[i] = pd.Categorical(dataset[i]).codes

dataset.tail()

X = np.zeros((13, len(dataset)))
y = np.array(dataset[dataset.columns[13]])

#assign values to the array
for i in range(len(dataset.columns)-1):
    X[i] = np.array(dataset[dataset.columns[i]])
X = X.transpose()

# How to handle the over-fitting problem?
# setting testing and training data limit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)



#training and predictions
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)



# predict test data
y_pred = classifier.predict(X_test)
print(y_pred)

# How to compare the designed machine learning models?
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
