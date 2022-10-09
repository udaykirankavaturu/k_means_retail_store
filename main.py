# Steps:
# 1. Read and understand data
# 2. Clean up data
# 3. Prepare data for modelling (STANDARDISE)
# 4. Modelling
# 5. Final analysis and reco


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# 1. Read and understand data
retail_df = pd.read_csv('Online_Retail.csv', sep=",",
                        encoding="ISO-8859-1", header=0)

# print first five records
print('first five records')
print(retail_df.head())

# print metrics of data
print('metrics of data')
print(retail_df.info())


# 2. Clean up data
# check for null values, sum them, check for % of null values, round to 2 decimal places
print(round(100*retail_df.isnull().sum()/len(retail_df)), 2)
# delete all rows with null values
retail_df = retail_df.dropna()
print('shape after dropping null values')
print(retail_df.shape)

# 3. Prepare the data
# R(Recency): number of days since last transaction
# F(Frequency): number of transactions in the data set
# M(Monetary value): total of amounts from transactions (revenue contributed by this customer)
# desired format: cust_id recency frequency monetary_value

# add new column for amount
retail_df['Monetary'] = retail_df['Quantity'] * retail_df['UnitPrice']
print(retail_df.head())

# MONETARY
monetary_df = retail_df.groupby('CustomerID')['Monetary'].sum()
monetary_df = monetary_df.reset_index()
print('after monetary step')
print(monetary_df.head())

# FREQUENCY
frequency_df = retail_df.groupby('CustomerID')['InvoiceNo'].count()
frequency_df = frequency_df.reset_index()
frequency_df.columns = ['CustomerID', 'Frequency']
print('after frequency step')
print(frequency_df.head())

# merge both data frames
monetary_df = pd.merge(monetary_df, frequency_df, on='CustomerID', how='inner')
print('after merge step')
print(monetary_df.head())


# RECENCY
# change format of invoice date to pandas datetime, currently it's object type
retail_df['InvoiceDate'] = pd.to_datetime(
    retail_df['InvoiceDate'], format='%d-%m-%Y %H:%M')
print(retail_df.info())  # check if invoice date is now datetime

# get max date
max_date = max(retail_df['InvoiceDate'])
print('max date in given data set', max_date)

# calculate difference
retail_df['diff'] = max_date - retail_df['InvoiceDate']
print('after diff step')
print(retail_df.head())

# group by recency
recency_df = retail_df.groupby('CustomerID')['diff'].min()
recency_df = recency_df.reset_index()
recency_df.columns = ['CustomerID', 'Recency']
print(recency_df.head())

# merge
monetary_df = pd.merge(monetary_df, recency_df, on='CustomerID', how='inner')
print('after merge step')
print(monetary_df.head())

# just use days from recency
monetary_df['Recency'] = monetary_df['Recency'].dt.days
print('after cleaning up recency')
print(monetary_df.head())
grouped_df = monetary_df

# 3. Prepare data for modelling (STANDARDISE)
# outlier treatment
plt.boxplot(grouped_df['Monetary'])
plt.show()
# rescaling
