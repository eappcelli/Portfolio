# # K Means Clustering Project 

# ## Import Libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data

college = pd.read_csv('College_Data', index_col=0)
college.info()
college.head()
college.describe()

# ## EDA

# ** Scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

sns.lmplot(x='Room.Board', y='Grad.Rate', data=college, hue='Private', fit_reg=False, palette='coolwarm', height=6, aspect=1)

# **Scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

sns.lmplot(x='Outstate', y='F.Undergrad', data=college, hue='Private', fit_reg=False, height=6, aspect=1)

# ** Stacked histogram showing Out of State Tuition based on the Private column using sns.FacetGrid **

g = sns.FacetGrid(college, hue='Private', palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=.5)

# Another method:

sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
college[college['Private']=='Yes']['Outstate'].plot(kind='hist', bins=25, alpha=0.5)
college[college['Private']=='No']['Outstate'].plot(kind='hist', bins=25, alpha=0.5)
plt.xlabel('Out of State Tuition')


# **Similar histogram for the Grad.Rate column.**


plt.figure(figsize=(12,6))
college[college['Private']=='Yes']['Grad.Rate'].plot(kind='hist', bins=20, alpha=0.5)
college[college['Private']=='No']['Grad.Rate'].plot(kind='hist', bins=20, alpha=0.5)
plt.xlabel('Graduation Rate')


# Correct graduation rate which is higher than 100%. **

college[college['Grad.Rate']>100]

college['Grad.Rate'] = college['Grad.Rate'].replace([118], 100)


# ## K Means Cluster Creation

# ** Import KMeans from SciKit Learn.**

from sklearn.cluster import KMeans

# ** Create an instance of a K Means model with 2 clusters.**

km = KMeans(n_clusters=2)

# **Fit the model.**

km.fit(college.drop('Private', axis=1))

# ** View cluster center vectors**

km.cluster_centers_
