# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:

To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import required libraries and load the salary dataset.

2.Select input feature (Level) and target variable (Salary).

3.Train the Decision Tree Regressor model using training data.

4.Predict salary values and evaluate model performance. 

## Program:
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SAADHANA A
RegisterNumber: 25018432 
*/

```
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"C:\Users\acer\Downloads\Salary.csv") 
print("Dataset Loaded Successfully\n")
print(data.head())

X = data.iloc[:, 1:2].values  
y = data.iloc[:, 2].values     


model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)

pred = model.predict([[6.5]])
print("\nPredicted Salary for Level 6.5:", pred)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y)
plt.plot(X_grid, model.predict(X_grid))
plt.title("Decision Tree Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
```

## Output:

![Screenshot_21-2-2026_213449_localhost](https://github.com/user-attachments/assets/bc1883fe-6bde-4e58-942a-6b60edc793c5)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
