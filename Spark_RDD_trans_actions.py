
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic('writefile', 'example2.txt', 'first\nsecond line\nthe third line\nthen a fourth line')


# In[2]:

from pyspark import SparkContext


# In[4]:

sc = SparkContext()


# In[6]:

text_rdd = sc.textFile('example2.txt')


# In[7]:

words = text_rdd.map(lambda line: line.split())


# In[8]:

words.collect()


# In[9]:

text_rdd.flatMap(lambda line: line.split()).collect()


# In[10]:

get_ipython().run_cell_magic('writefile', 'services.txt', '#EventId    Timestamp    Customer   State    ServiceID    Amount\n201       10/13/2017      100       NY       131          100.00\n204       10/18/2017      700       TX       129          450.00\n202       10/15/2017      203       CA       121          200.00\n206       10/19/2017      202       CA       131          500.00\n203       10/17/2017      101       NY       173          750.00\n205       10/19/2017      202       TX       121          200.00')


# In[11]:

services = sc.textFile('services.txt')


# In[12]:

services.take(2)


# In[14]:

services.map(lambda line: line.split()).take(3)


# In[16]:

clean = services.map(lambda line: line[1:] if line[0] == '#' else line)


# In[17]:

clean = clean.map(lambda line: line.split())


# In[18]:

clean.collect()


# In[21]:

pairs = clean.map(lambda lst: (lst[3], lst[-1]))


# In[24]:

rekey = pairs.reduceByKey(lambda amt1, amt2 : float(amt1) + float(amt2))


# In[25]:

rekey.collect()


# In[26]:

clean.collect()


# In[27]:

step1 = clean.map(lambda list: (list[3], list[-1]))


# In[28]:

step2 = step1.reduceByKey(lambda amt1, amt2 : float(amt1) + float(amt2))


# In[31]:

step3 = step2.filter(lambda x: not x[0]=='State')


# In[33]:

step4 = step3.sortBy(lambda stAmt: stAmt[1], ascending=False)


# In[34]:

step4.collect()


# In[35]:

x = ['ID', 'State', 'Amount']


# In[36]:

def func1(lst):
    return lst[-1]


# In[37]:

def func2(id_st_amt):
    #unpack values
    (Id, st, amt) = id_st_amt
    return amt


# In[38]:

func1(x)


# In[39]:

func2(x)


# In[ ]:



