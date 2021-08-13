import numpy
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# return slope and y intercept of best fit line that covers coordinates in data points
def create(datapoints):
    '''->Create x matrix from datapoints
         -> Create y matrix from datapoints
          ->find x transpose(xT)

          ->multiply xT with x (B)

         ->mutiply xT with y (C)

         ->find B^-1

         ->Multiply B^-1 with C'''

    x=[[c[0],1] for c in datapoints]
    print(x)
    y=[[c[1]] for c in datapoints ]
    x=numpy.array(x)
    y=numpy.array(y)
    xt=x.T
    b=numpy.dot(xt,x)
    c=numpy.dot(xt,y)
    b=numpy.linalg.inv(b)
    return numpy.dot(b,c)

#Create dataframe  from student-mat.csv file



data = pd.read_csv("income.data.csv", sep=",")

#Extract required attrbutes
data = data[["income","happiness"]]
#print(data)
#Declare data to predict
predict = "happiness"

# X is an 2d array where each  1d array consists  .
X = np.array(data.drop([predict], 1))
print(X)
#
y = np.array(data[predict])

#
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3)

x_train=list(x_train)
for i in range(len(x_train)):
    x_train[i]=x_train[i][0]


x_train=list(x_train)
y_train=list(y_train)
y_test=list(y_test)

for i in range(len(x_test)):
    x_test[i]=x_test[i][0]


datapoints=[]
for i in range(len(x_train)):

    datapoints.append([x_train[i],y_train[i]])
#x=datapoints[0][0]
print(datapoints)



line=create(datapoints)
line=list(line)
slope=list(line[0])[0]
constant=list(line[0])[0]
print(slope)

print(constant)



for i in range(len(x_test)):
    prediction=(slope*x_test[i])+constant

    print(prediction, x_test[i], y_test[i])










