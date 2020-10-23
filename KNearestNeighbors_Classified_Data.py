# # K Nearest Neighbors Project 

# ## Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

df = pd.read_csv('KNN_Project_Data')

df.head()
sns.pairplot(df, hue='TARGET CLASS')

# # Standardize the Variables

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(df.drop('TARGET CLASS', axis=1))
df_scaledvar = scalar.transform(df.drop('TARGET CLASS', axis=1))

df_scaled = pd.DataFrame(df_scaledvar, columns=df.columns[:-1])
df_scaled.head()


# # Train Test Split

from sklearn.model_selection import train_test_split

X = df_scaled
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# # Using KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# # Predictions and Evaluations

pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# # Choosing a K Value

# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list.**
error_rate=[]

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)    
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# **Plot Error Rate vs K**

plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
plt.plot(np.arange(1, 51, 1), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value

knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)    
pred = knn.predict(X_test)
error_rate.append(np.mean(pred != y_test))
    
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

