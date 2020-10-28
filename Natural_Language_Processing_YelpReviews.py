# # Natural Language Processing Project

# ## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## The Data

yelp = pd.read_csv('yelp.csv')

yelp.head()
yelp.info()
yelp.describe()


# **Create a new column called "text length" which is the number of words in the text column.**

yelp['text length']= yelp['text'].apply(len)

# # EDA

# **Grid of 5 histograms of text length based off of the star ratings**

g = sns.FacetGrid(yelp, col='stars')
g.map(sns.distplot, 'text length', kde=False, bins=10)
sns.set_style('whitegrid')


# **Boxplot of text length for each star category.**

sns.boxplot('stars', 'text length', data=yelp, palette = 'rainbow')


# **Countplot of the number of occurrences for each type of star rating.**

sns.countplot(yelp['stars'], palette = 'rainbow')

# ** Heatmap showing correlations between numerical columns**

yelp.groupby(['stars']).mean()

yelp.groupby(['stars']).mean().corr()

sns.heatmap(yelp.groupby(['stars']).mean().corr(), cmap='coolwarm', annot=True)

# ## NLP Classification 
 
# **Dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars']==5)]

# ** Create two objects X and y: features and target/labels**

X = yelp_class['text']
y=yelp_class['stars']


# **CountVectorizer **

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(X)


# ## Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Classifier Model

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# # Using Text Processing
 
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline


# ** Pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**

pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('classifier', MultinomialNB())
])


# ### Train Test Split

X = yelp_class['text']
y=yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# **Fit the pipeline to the training data.**

pipe.fit(X_train, y_train)


# ### Predictions and Evaluation

predictions = pipe.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
