import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import sklearn # machine learning library
from sklearn import linear_model # linear regression
from sklearn.utils import shuffle # shuffle data

data = pd.read_csv("student-mat.csv", sep=";") # read data

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']] # features

predict = "G3" # what we want to predict

x = np.array(data.drop([predict], 1)) # features
y = np.array(data[predict]) # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) # split data into training and testing

linear = linear_model.LinearRegression() # create linear regression model

linear.fit(x_train, y_train) # train the model
acc = linear.score(x_test, y_test) # accuracy
print('Accuracy: ',acc) 

print('Coefficient: ', linear.coef_) # m
print('Intercept: ', linear.intercept_) # y = mx + b

predictions = linear.predict(x_test) # predictions

for x in range(len(predictions)): # print predictions
    print(predictions[x], x_test[x], y_test[x]) # prediction, features, label