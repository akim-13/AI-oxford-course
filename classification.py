import numpy as np 
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import tree, datasets
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import plot_confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X)
print(y)

col_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
iris_df = pd.DataFrame(X, columns=col_names)
plt.scatter(x=iris_df['Sepal length'], y=iris_df['Sepal width'], c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_combo_train_val = X_train
y_combo_train_val = y_train

kf = KFold(n_splits=2)
kf.get_n_splits(X_combo_train_val)
print(kf)

k = 2
KNC = KNeighborsClassifier(n_neighbors=k)

for train_index, val_index in kf.split(X_combo_train_val):
    X_fold, X_val = X_combo_train_val[train_index], X_combo_train_val[val_index]
    y_fold, y_val = y_combo_train_val[train_index], y_combo_train_val[val_index]
    KNC.fit(X_fold, y_fold)
    plot_confusion_matrix(KNC, X_val, y_val)
    plot_confusion_matrix(KNC, X_test, y_test)
    plt.show()

# plt.show()
