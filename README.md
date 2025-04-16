# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values. 


## Program:


Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: CHITTOOR SARAVANA MRUDHULA


RegisterNumber:  212224040056


```
import pandas as pd


data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = df.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])

```


## Output:

![image](https://github.com/user-attachments/assets/dce67eca-187f-408e-a865-f1f400138508)

![image](https://github.com/user-attachments/assets/7f9bb122-658e-485c-8ccc-2c08ac6ffeb3)

![image](https://github.com/user-attachments/assets/bdf2ecc0-bf60-4f8a-9540-38e27e390648)

![image](https://github.com/user-attachments/assets/d6e6df60-8dde-4147-9dd3-12ee4d391286)

![image](https://github.com/user-attachments/assets/cec12c92-3a14-4654-a9e9-0a9d43e451f3)

![image](https://github.com/user-attachments/assets/2a824a8f-8796-4d07-b793-0303a564185e)

![image](https://github.com/user-attachments/assets/57bf7a91-0932-4dc6-977f-e7eb33c2449c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
