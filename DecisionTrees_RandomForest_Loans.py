# # Random Forest Project 

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data

loans = pd.read_csv('loan_data.csv')
loans.info()
loans.describe()
loans.head()

# # Exploratory Data Analysis

# ** Histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=30, alpha=0.5, label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(bins=30, alpha=0.5, label='Credit.Policy=0')

plt.legend()
plt.xlabel('FICO')

# ** Similar figure, this time by the not.fully.paid column**

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, color = 'red', label='Not Fully Paid = 1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, color = 'blue', label='Not Fully Paid = 0')

plt.legend()
plt.xlabel('FICO')


# ** Countplot showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

plt.figure(figsize=(12,6))
sns.countplot('purpose', hue='not.fully.paid', data=loans)
plt.tight_layout()

# ** Trend between FICO score and interest rate**

sns.jointplot('fico', 'int.rate', data=loans)

# ** lmplots to see if the trend differed between not.fully.paid and credit.policy**

sns.lmplot(x="fico", y="int.rate", hue='credit.policy', col="not.fully.paid", data=loans)

# # Setting up the Data

loans.info()

# ## Categorical Features

cat_feats=['purpose']

final_data=pd.get_dummies(loans, columns=cat_feats, drop_first=True)
final_data.head()


# ## Train Test Split

from sklearn.model_selection import train_test_split

X = final_data.drop(['not.fully.paid'], axis=1)
y= loans['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Training a Decision Tree Model

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)


# ## Predictions and Evaluation of Decision Tree

from sklearn.metrics import classification_report, confusion_matrix
pred=dtree.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# ## Training the Random Forest model

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)


# ## Predictions and Evaluation

pred_r = rfc.predict(X_test)

print(classification_report(y_test,pred_r))
print(confusion_matrix(y_test,pred_r))

