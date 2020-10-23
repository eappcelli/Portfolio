
# ## Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ## Get the Data
# **Read in the advertising.csv file 

ad_data = pd.read_csv('advertising.csv')

ad_data.head()
ad_data.info()
ad_data.describe()

# ## Exploratory Data Analysis

# ** Create a histogram of the Age**

sns.distplot(ad_data['Age'])
sns.set_style('whitegrid')

# **Create a jointplot showing Area Income versus Age.**

sns.jointplot('Age', 'Area Income', data=ad_data)

# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

sns.jointplot('Age', 'Daily Time Spent on Site', data=ad_data, kind='kde', color='red')

# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

sns.jointplot('Daily Time Spent on Site', 'Daily Internet Usage', data=ad_data, color='green', xlim=(20,100), ylim=(50,300), s=10)

# ** Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

sns.pairplot(ad_data, hue='Clicked on Ad', diag_kind='hist')


# # Logistic Regression

# ** Split the data into training set and testing set using train_test_split**


from sklearn.model_selection import train_test_split

ad_data.columns
X= ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y=ad_data['Clicked on Ad']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# ** Train and fit a logistic regression model on the training set.**

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = LogisticRegression(solver='lbfgs', max_iter=400)
lr.fit(X_train, y_train)

# ## Predictions and Evaluations

predict = lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))
