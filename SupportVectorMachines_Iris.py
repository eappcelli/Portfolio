# # Support Vector Machines Project 

# ## Get the data

iris = sns.load_dataset('iris')

# **Import libraries **

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Exploratory Data Analysis

iris.head()

# **Pair plot by species.**

sns.pairplot(iris, hue='species')

# **Kde plot of sepal length versus sepal width for setosa species of flower.**

setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)


# # Train Test Split

from sklearn.model_selection import train_test_split


X=iris.drop('species', axis=1)
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Train a Model

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)


# ## Model Evaluation

preds = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


# ## Gridsearch Practice

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000], 'gamma': [1, 0.1,0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

grid.best_params_

preds = grid.predict(X_test)

print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
