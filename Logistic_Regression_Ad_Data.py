#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[3]:


ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[4]:


ad_data.head()


# In[40]:





# ** Use info and describe() on ad_data**

# In[5]:


ad_data.info()


# In[41]:





# In[6]:


ad_data.describe()


# In[ ]:





# In[ ]:





# In[42]:





# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[9]:


sns.distplot(ad_data['Age'])
sns.set_style('whitegrid')


# In[48]:





# **Create a jointplot showing Area Income versus Age.**

# In[11]:


sns.jointplot('Age', 'Area Income', data=ad_data)


# In[64]:





# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[14]:


sns.jointplot('Age', 'Daily Time Spent on Site', data=ad_data, kind='kde', color='red')


# In[66]:





# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[20]:


sns.jointplot('Daily Time Spent on Site', 'Daily Internet Usage', data=ad_data, color='green', xlim=(20,100), ylim=(50,300), s=10)


# In[72]:





# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[53]:


sns.pairplot(ad_data, hue='Clicked on Ad', diag_kind='hist')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[22]:


ad_data.columns


# In[25]:


ad_data['Country'].nunique()


# In[23]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[56]:


X= ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y=ad_data['Clicked on Ad']


# In[38]:


X.head()
y.head()


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ** Train and fit a logistic regression model on the training set.**

# In[40]:


from sklearn.linear_model import LogisticRegression


# In[58]:


lr = LogisticRegression()


# In[59]:


lr = LogisticRegression(solver='lbfgs', max_iter=400)


# In[60]:


lr.fit(X_train, y_train)


# In[92]:





# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[61]:


predict = lr.predict(X_test)


# ** Create a classification report for the model.**

# In[62]:


from sklearn.metrics import classification_report


# In[63]:


print(classification_report(y_test, predict))


# In[96]:





# ## Great Job!
