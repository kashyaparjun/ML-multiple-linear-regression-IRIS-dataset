import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris_pyth.csv').values
x = dataset[:, :-1]
y = dataset[:, 4]

#Encoding SPecies
for i in range(len(y)):
    if(y[i]=='virginica'):
        y[i] = 3
    elif(y[i]=='versicolor'):
        y[i] = 2
    else:
        y[i] = 1
         
    
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Fitting Multiple regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
ty = []
tym = []
k = open('oppn.csv','w+')
k = open('oppn.csv','a')
for i in range(len(y_pred)):
    kg = [str(round(y_pred[i])),str(y_test[i])]
    ty.append(kg)
    
for i in range(len(ty)):
    l = ty[i]
    h = float(l[0])
    pl = ""
    klt = ""
    jk = []
    if(h==1.0):
        pl = "setosa"
    elif(h==2.0):
        pl = "versicolor"
    else:
        pl = "virginica"
    h = float(l[1])
    if(h==1.0):
        klt = "setosa"
    elif(h==2.0):
        klt = "versicolor"
    else:
        klt = "virginica"
    jk.append(pl)
    jk.append(klt)
    tym.append(jk)
        
    
for i in range(len(tym)):
    k.write(str(tym[i])+"\n")
k.flush()
k.close()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_test, color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





