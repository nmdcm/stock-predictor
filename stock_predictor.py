import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

dates = []
prices = []
#Read the CSV file and preprocess the data
df =  pd.read_csv('GOOGL.csv')
df['Day'] = pd.DatetimeIndex(df['Date']).day
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Year'] = pd.DatetimeIndex(df['Date']).year
df = df.loc[(df['Year'] == 2019) & (df['Month'] == 5)]
dates = df['Day'].tolist()
prices = df['Close'].tolist()

#Fit the data to three types of SVR
dates = np.reshape(dates,(len(dates), 1))

svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
svr_rbf.fit(dates, prices)
svr_lin.fit(dates, prices)
svr_poly.fit(dates, prices)

#Plot the data and them model
plt.scatter(dates, prices, color= 'black', label= 'Data')
plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model')
plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model')

#Plot predicted values for the 29th
plt.scatter([[29]], svr_rbf.predict([[29]])[0], color= 'red', label= 'Predicted RBF')
plt.scatter([[29]], svr_lin.predict([[29]])[0], color= 'green', label= 'Predicted Linear')
plt.scatter([[29]], svr_poly.predict([[29]])[0], color= 'blue', label= 'Predicted Polynomial')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

#Print predicted prices for the 29th
print('Predicted RBF:',svr_rbf.predict([[29]])[0])
print('Predicted Linear:',svr_lin.predict(29)[0])
print('Predicted Polynomial:',svr_poly.predict(29)[0])
