import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
#Create dataframe  from student-mat.csv file
data = pd.read_csv("student-mat.csv", sep=";")



data = pd.read_csv("student-mat.csv", sep=";")

#Extract required attrbutes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data)
#Declare data to predict
predict = "G3"

# X is an 2d array where each  1d array consists  .
X = np.array(data.drop([predict], 1))
print(X)
#
y = np.array(data[predict])

#
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.6)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# Since our data is seperated by semicolons we need to do sep=";"