import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sep():
    print('====================================================')

data, labels = load_iris(return_X_y=True)
print(data, labels)
sep()

data = pd.DataFrame(data)
labels = pd.DataFrame(labels, columns=['labels'])

print(f'X:\n{data}')
sep()

print(f'Y:\n{labels}')
sep()

unique_labels = pd.DataFrame(labels['labels'].unique())
print(f'Unique labels:\n{unique_labels}')
sep()

unique_labels['count'] = unique_labels[0].map(labels['labels'].value_counts())
print(f'Mapped unique labels:\n{unique_labels}')
sep()

plt.bar(['iris', 'petunia', 'setosa'], unique_labels['count'])

# Allocate 70% of dataset for training and 30% for testing.
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.7)

lr = LogisticRegression(multi_class='auto')
lr.fit(train_data, np.ravel(train_labels))

prediction = lr.predict(test_data)
print(f'Prediction: {prediction}')
print(f'Accuracy of prediction: {accuracy_score(test_labels, prediction)}')

plt.show()
