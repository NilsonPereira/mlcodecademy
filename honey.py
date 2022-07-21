import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")

prod_per_year = df.groupby('year').totalprod.mean().reset_index()
#print(prod_per_year)

X = prod_per_year['year']
X = X.values.reshape(-1, 1)

y = prod_per_year['totalprod']

line_fitter = linear_model.LinearRegression()
line_fitter.fit(X, y)
y_predict = line_fitter.predict(X)

print(line_fitter.coef_)
print(line_fitter.intercept_)

X_future = np.array(range(1998,2051))
X_future = X_future.reshape(-1, 1)

future_predict = line_fitter.predict(X_future)

print(X_future[-2:])
print(future_predict[-2:])

plt.scatter(X, y)
#plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.show()



print("OK")