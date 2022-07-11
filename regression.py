import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def print_regression(name):
    print(f'\n{name} Regression:')
    print(f'1) Mean squared error: {mse}')
    print(f'2) Mean absolute error: {mae}')
    print(f'3) R^2 value: {r_2}')

# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# [LABEL TO PREDICT] MEDV - Median value of owner-occupied homes in $1000's
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'MEDV' ]

data = pd.read_csv('boston_house.csv', header=None, delimiter=r'\s+', names=col_names)
print(data.head(5))

data_features = data.drop(columns=['MEDV'])
label_to_predict = data['MEDV']

fig = plt.figure(figsize=(10, 10))
plt.scatter(x=data_features['RM'],y=data_features['AGE'])
plt.xlabel('RM')
plt.ylabel('AGE')
plt.title('Graph of RM against AGE')

X = data_features
y = label_to_predict
test_size = 0.2

# Dividing dataset into 80% for training and 20% for testing.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
# Dividing the training part (80%) into 80% training and 20% validation.
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size, random_state=42)

# Scaling the data using Z distribution.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

# Accuracy.
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
score = LR_model.score(X_test, y_test) 
print(f'Score: {score}')

# Features
fig = plt.figure(figsize=(10, 10))
plt.barh(list(X.columns), LR_model.coef_)
plt.xlabel("Feature Importance")


# Linear regression.
y_predict = LR_model.predict(X_test)
mse = np.round(mean_squared_error(y_test, y_predict), 2)
mae = np.round(mean_absolute_error(y_test, y_predict), 2)
r_2 = np.round(r2_score(y_test, y_predict), 2)
print_regression('Linear')
y_true = y_test.reset_index(drop=True)

fig = plt.figure(figsize=(10, 10))
plt.plot(y_predict, label='Linear: Predicted Price')
plt.plot(y_true, label='Linear: True Price')
plt.legend()


# Polynomial regression.
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)
poly_model = LinearRegression().fit(poly_X_train, y_train)
y_predict = poly_model.predict(poly_X_test)

mse = np.round(mean_squared_error(y_test, y_predict), 2)
mae = np.round(mean_absolute_error(y_test, y_predict), 2)
r_2 = np.round(r2_score(y_test, y_predict), 2)
print_regression('Polynomial')

fig = plt.figure(figsize=(10, 10))
plt.plot(y_predict, label='Polynomial: Predicted Price')
plt.plot(y_test.reset_index(drop=True), label='Polynomial: True Price')
plt.legend()
plt.show()
