#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # K Means Clustering Project 
# 
# For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.
# 
# ___
# It is **very important to note, we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm.** 
# 
# When using the Kmeans algorithm under normal circumstances, it is because you don't have labels. In this case we will use the labels to try to get an idea of how well the algorithm performed, but you won't usually do this for Kmeans, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting!.
# ___
# 
# ## The Data
# 
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# ## Import Libraries
# 
# ** Import the libraries you usually use for data analysis.**

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data

# ** Read in the College_Data file using read_csv. Figure out how to set the first column as the index.**

# In[10]:


college = pd.read_csv('College_Data', index_col=0)


# In[11]:


college.info()


# **Check the head of the data**

# In[12]:


college.head()


# ** Check the info() and describe() methods on the data.**

# In[13]:


college.info()


# In[106]:





# In[14]:


college.describe()


# In[107]:





# ## EDA
# 
# It's time to create some data visualizations!
# 
# ** Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

# In[85]:


sns.lmplot(x='Room.Board', y='Grad.Rate', data=college, hue='Private', fit_reg=False, palette='coolwarm', height=6, aspect=1)


# In[111]:





# **Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

# In[86]:


sns.lmplot(x='Outstate', y='F.Undergrad', data=college, hue='Private', fit_reg=False, height=6, aspect=1)


# In[112]:





# ** Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **

# In[90]:


g = sns.FacetGrid(college, hue='Private', palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=.5)


# In[43]:


sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
college[college['Private']=='Yes']['Outstate'].plot(kind='hist', bins=25, alpha=0.5)
college[college['Private']=='No']['Outstate'].plot(kind='hist', bins=25, alpha=0.5)
plt.xlabel('Out of State Tuition')


# In[45]:


plt.figure(figsize=(12,6))
college[college['Private']=='Yes']['Grad.Rate'].plot(kind='hist', bins=20, alpha=0.5)
college[college['Private']=='No']['Grad.Rate'].plot(kind='hist', bins=20, alpha=0.5)
plt.xlabel('Graduation Rate')


# In[109]:





# **Create a similar histogram for the Grad.Rate column.**

# In[91]:


college[college['Grad.Rate']>100]


# In[56]:


college[college['Grad.Rate'] == college['Grad.Rate'].max()]


# ** Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?**

# In[113]:





# ** Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.**

# In[60]:


college['Grad.Rate'] = college['Grad.Rate'].replace([118], 100)


# In[61]:


plt.figure(figsize=(12,6))
college[college['Private']=='Yes']['Grad.Rate'].plot(kind='hist', bins=20, alpha=0.5)
college[college['Private']=='No']['Grad.Rate'].plot(kind='hist', bins=20, alpha=0.5)
plt.xlabel('Graduation Rate')


# In[94]:





# In[62]:


college.describe()


# In[95]:





# ## K Means Cluster Creation
# 
# Now it is time to create the Cluster labels!
# 
# ** Import KMeans from SciKit Learn.**

# In[64]:


from sklearn.cluster import KMeans


# ** Create an instance of a K Means model with 2 clusters.**

# In[65]:


km = KMeans(n_clusters=2)


# **Fit the model to all the data except for the Private label.**

# In[68]:


km.fit(college.drop('Private', axis=1))


# ** What are the cluster center vectors?**

# In[69]:


km.cluster_centers_


# In[117]:





# ## Evaluation
# 
# There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
# 
# ** Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**

# In[79]:


college['Cluster'] = np.where(college['Private'] == 'Yes', 1, 0)


# In[80]:


college.head()


# In[122]:





# ** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# In[81]:


from sklearn.metrics import classification_report, confusion_matrix


# In[82]:


print(confusion_matrix(college['Cluster'], km.labels_))


# In[83]:


print(classification_report(college['Cluster'], km.labels_))


# In[123]:





# Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! Hopefully you can begin to see how K Means is useful for clustering un-labeled data!
# 
# ## Great Job!
