# Python testing
#
# side = 5
# print('Welcome to ML Word')
# for x in list(range(side)) + list(reversed(range(side-1))):
#     print('{: <{w1}}{:*<{w2}}'.format('', '', w1=side-x-1, w2=x*2+1))
#
# print('Asela Wijesinghe - AS2015606')
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils


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

# configs
epochs = 100
batch_size = 88
output_classes = 2

y_train = np_utils.to_categorical(y_train, output_classes)
y_test = np_utils.to_categorical(y_test, output_classes)



# Set up the logistic regression model
model = Sequential()
model.add(Dense(2, input_dim=(13), kernel_initializer='normal', activation='softmax'))
model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print(model.summary())
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

evaluation = model.evaluate(X_test, y_test, verbose=1)
print(evaluation[1])
