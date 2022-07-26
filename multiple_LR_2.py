import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

maht = pd.read_csv("data/manhattan.csv")

#print(df.head())
#print(df[['min_to_subway','neighborhood']])

df = pd.DataFrame(maht)

#ylist = ['bedrooms','bathrooms','size_sqft','min_to_subway','floor','building_age_yrs','no_fee','has_roofdeck','has_washer_dryer','has_doorman','has_elevator','has_dishwasher','has_patio','has_gym']

x = df[['bedrooms','bathrooms','size_sqft','min_to_subway','floor','building_age_yrs','no_fee','has_roofdeck','has_washer_dryer','has_doorman','has_elevator','has_dishwasher','has_patio','has_gym']]
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

mlr = LinearRegression()
mlr.fit(x_train, y_train) 

print("Fit coefs: {}".format(mlr.coef_))
print("Fit intercept: {}".format(mlr.intercept_))

y_predict = mlr.predict(x_test)

sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]

predict = mlr.predict(sonny_apartment)

print("Predicted rent: $%.2f" % predict)

print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))

# Create a scatter plot
#plt.scatter(y_test, y_predict, alpha=0.4)
#plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)
plt.scatter(df[['min_to_subway']], df[['rent']], alpha=0.4)

# Create x-axis label and y-axis label
#plt.xlabel("Test values")
#plt.ylabel("Predicted values")

# Create a title
#plt.title("Multiple linear regression")

plt.show()



print("OK")