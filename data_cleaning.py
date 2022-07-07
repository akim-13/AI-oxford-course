import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(6, 6))
df[df > 0.9] = np.nan
print(df)
print()
print(df.isnull().sum())
print()
print(df.isnull().sum().sum())
print()
# Replace NaN with the mean in the first column.
df[0].fillna(df[0].mean(), inplace = True)
print(df)
