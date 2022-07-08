import pandas as pd
import numpy as nu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans

def convert_to_bin(col_name, customer_data):
    one_hot = pd.get_dummies(customer_data[col_name])
    customer_data.drop(col_name, axis = 1, inplace = True)
    customer_data = one_hot.join(customer_data)
    return customer_data

def sep():
    print('========================================================================')

customer_data = pd.read_csv('retail_data.csv')
print(customer_data)
sep()

# Check if there are any missing values.
print(customer_data.isnull().sum().sum())
sep()

# Converting some columns to binary.
customer_data = convert_to_bin('Gender', customer_data)
print(customer_data[:5])
sep()
customer_data = convert_to_bin('Married', customer_data)
print(customer_data[:5])
sep()

# Standardizing the data
standardized_customer_data = preprocessing.scale(customer_data)
standardized_customer_data_df = pd.DataFrame(standardized_customer_data, columns = customer_data.columns)
print(standardized_customer_data_df)
sep()

# Train the model
kmeans = KMeans(n_clusters=4)
kmeans.fit(standardized_customer_data_df)
clusters = kmeans.fit_predict(standardized_customer_data_df)
print(clusters[:500])
sep()
print(customer_data[clusters==2][:10])

# Plot the data
plt.figure(figsize=(10, 10))
plt.scatter(customer_data[clusters == 0]['Age'], customer_data[clusters == 0]['Salary'], s = 15, c = 'red', alpha = .5)
plt.scatter(customer_data[clusters == 1]['Age'], customer_data[clusters == 1]['Salary'], s = 15, c = 'orange', alpha = .5)
plt.scatter(customer_data[clusters == 2]['Age'], customer_data[clusters == 2]['Salary'], s = 15, c = 'green', alpha = .5)
plt.scatter(customer_data[clusters == 3]['Age'], customer_data[clusters == 3]['Salary'], s = 15, c = 'blue', alpha = .5)

plt.figure(figsize=(7, 7))
plt.scatter(customer_data[clusters == 0]['Female'], customer_data[clusters == 0]['Salary'], s = 15, c = 'red', alpha = .5)
plt.scatter(customer_data[clusters == 1]['Female'], customer_data[clusters == 1]['Salary'], s = 15, c = 'orange', alpha = .5)
plt.scatter(customer_data[clusters == 2]['Female'], customer_data[clusters == 2]['Salary'], s = 15, c = 'green', alpha = .5)
plt.scatter(customer_data[clusters == 3]['Female'], customer_data[clusters == 3]['Salary'], s = 15, c = 'blue', alpha = .5)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(20, 20)
ax.set_xlabel('Annual Spend')
ax.set_ylabel('Salary')
ax.set_zlabel('Age')
plt.scatter(customer_data[clusters == 0]['Annual Spend'], customer_data[clusters == 0]['Salary'], s = 15, c = 'red', alpha = .5)
plt.scatter(customer_data[clusters == 1]['Annual Spend'], customer_data[clusters == 1]['Salary'], s = 15, c = 'orange', alpha = .5)
plt.scatter(customer_data[clusters == 2]['Annual Spend'], customer_data[clusters == 2]['Salary'], s = 15, c = 'green', alpha = .5)
plt.scatter(customer_data[clusters == 3]['Annual Spend'], customer_data[clusters == 3]['Salary'], s = 15, c = 'blue', alpha = .5)

customer_data.plot.scatter('Age', 'Salary')
standardized_customer_data_df.plot.scatter('Age', 'Salary')
plt.show()
