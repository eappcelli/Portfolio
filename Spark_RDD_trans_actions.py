
# Create example data:

get_ipython().run_cell_magic('writefile', 'example2.txt', 'first\nsecond line\nthe third line\nthen a fourth line')


# Import SparkContext

from pyspark import SparkContext

sc = SparkContext()


# Map and FlatMap transformations:

text_rdd = sc.textFile('example2.txt')

words = text_rdd.map(lambda line: line.split())

words.collect()

text_rdd.flatMap(lambda line: line.split()).collect()


# Create example data:

get_ipython().run_cell_magic('writefile', 'services.txt', '#EventId    Timestamp    Customer   State    ServiceID    Amount\n201       10/13/2017      100       NY       131          100.00\n204       10/18/2017      700       TX       129          450.00\n202       10/15/2017      203       CA       121          200.00\n206       10/19/2017      202       CA       131          500.00\n203       10/17/2017      101       NY       173          750.00\n205       10/19/2017      202       TX       121          200.00')


# RDDs and Key Value Pairs:

services = sc.textFile('services.txt')

services.map(lambda line: line.split()).take(3)

clean = services.map(lambda line: line[1:] if line[0] == '#' else line)

clean = clean.map(lambda line: line.split())

clean.collect()


#Using Key Value Pairs for Operations:

#Grab state and amounts
step1 = clean.map(lambda list: (list[3], list[-1]))

# Add them
step2 = step1.reduceByKey(lambda amt1, amt2 : float(amt1) + float(amt2))


# Get rid of ('State', 'Amount')
step3 = step2.filter(lambda x: not x[0]=='State')

#Sort by amount value
step4 = step3.sortBy(lambda stAmt: stAmt[1], ascending=False)

# ACTION
step4.collect()


#Example of tuple unpacking for readability:

x = ['ID', 'State', 'Amount']


# Grab 'amount' from x Without tuple unpacking:

def func1(lst):
    return lst[-1]


# Grab 'amount' from x USING tuple unpacking:

def func2(id_st_amt):
    #unpack values
    (Id, st, amt) = id_st_amt
    return amt
