import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

customers = pd.read_csv('Ecommerce Customers')


# **Check the head of customers, and check out its info() and describe() methods.**

customers.head()
customers.info()
customers.describe()

### Exploratory Data Analysis

sns.jointplot(['Time on Website'], ['Yearly Amount Spent'], data=customers)
sns.set_style('whitegrid')
sns.jointplot(['Time on App'], ['Yearly Amount Spent'], data=customers)
sns.jointplot(['Time on App'], ['Yearly Amount Spent'], data=customers, kind='hex')
sns.pairplot(customers)
sns.lmplot('Length of Membership', 'Yearly Amount Spent', data=customers)

# ## Training and Testing Data

from sklearn.model_selection import train_test_split
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers[['Yearly Amount Spent']]


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets.**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ## Training the Model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_

# ## Predicting Test Data
predict=lm.predict(X_test)
plt.scatter(y_test, predict)

# ## Evaluating the Model
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('MSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))

# ## Residuals

sns.distplot(y_test - predict, bins=50)

# ## Model Coefficients 

lm.coef_.transpose()
coeff_df = pd.DataFrame(lm.coef_.transpose(),index=X.columns,columns=['Coefficient'])
coeff_df

