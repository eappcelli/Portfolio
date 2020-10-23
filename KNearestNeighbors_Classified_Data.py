#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # K Nearest Neighbors Project 
# 
# Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[3]:


df = pd.read_csv('KNN_Project_Data')


# **Check the head of the dataframe.**

# In[4]:


df.head()


# In[23]:





# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[5]:


sns.pairplot(df, hue='TARGET CLASS')


# In[4]:





# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[8]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[9]:


scalar = StandardScaler()


# ** Fit scaler to the features.**

# In[11]:


scalar.fit(df.drop('TARGET CLASS', axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[12]:


df_scaledvar = scalar.transform(df.drop('TARGET CLASS', axis=1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[14]:


df_scaled = pd.DataFrame(df_scaledvar, columns=df.columns[:-1])
df_scaled.head()


# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X = df_scaled
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**

# In[18]:


knn = KNeighborsClassifier(n_neighbors=1)


# **Fit this KNN model to the training data.**

# In[19]:


knn.fit(X_train, y_train)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[22]:


pred = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[21]:


from sklearn.metrics import classification_report, confusion_matrix


# In[23]:


print(confusion_matrix(y_test, pred))


# In[24]:


print(classification_report(y_test, pred))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[30]:


error_rate=[]

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)    
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[43]:


np.shape(error_rate)


# In[46]:


error_rate


# **Now create the following plot using the information from your for loop.**

# In[62]:


plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
plt.plot(np.arange(1, 51, 1), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[20]:





# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# In[64]:


knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)    
pred = knn.predict(X_test)
error_rate.append(np.mean(pred != y_test))
    
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[21]:





# # Great Job!
