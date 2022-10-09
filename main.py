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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
# plt.boxplot(grouped_df['Recency'])
# plt.show()
# remove outlier data after discussion with business
# because we don't know which data point looks extremely unusual


# rescaling - bring all dimensions to one scale
# get subset of main data frame to remove customer id
rfm_df = grouped_df[['Recency', 'Frequency', 'Monetary']]
print(rfm_df.head())

# use a standard scaler to bring all dimensions to one scale
scaler = StandardScaler()

rfm_df_scaled = scaler.fit_transform(rfm_df)
print('after standard scaling')
print(rfm_df_scaled)

# checking if the data frame is suitable for clustering or not using HOPKINS STATISTIC


def hopkins(X):
    d = X.shape[1]
    # d = len(vars) # columns
    n = len(X)  # rows
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(
            X, axis=0), d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(
            X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H


# First convert the numpy array that you have to a dataframe
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Recency', 'Frequency', 'Monetary']

# Use the Hopkins Statistic function by passing the above dataframe as a paramter.

# The Hopkins statistic (introduced by Brian Hopkins and John Gordon Skellam) is a way of measuring the cluster tendency
# of a data set.[1] It belongs to the family of sparse sampling tests.
# It acts as a statistical hypothesis test where the null hypothesis is that the data is generated
# by a Poisson point process and are thus uniformly randomly distributed.[2] A value close to 1
# tends to indicate the data is highly clustered, random data will tend to result in values around 0.5,
# and uniformly distributed data will tend to result in values close to 0.[3]
print('hopkins statistic value')
print(hopkins(rfm_df_scaled))


# run the k-means algorithm on the dataset
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)
print('after clustering')
print(kmeans.labels_)


# how to determine the number of clusters
# ELBOW CURVE METHOD
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    # run the clustering
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)

    # get the ssds and append
    ssd.append(kmeans.inertia_)

# plot the SSDs (sum of squared distances)
# plt.plot(ssd)
# plt.show()

# observe the drop between number of clusters
# if the drop is significant it's good, once drop becomes less significant we can stop
# in this example, ssd drops about 50% from number of clusters from 2 to 3, but not so much after
# so we can optimally choose to have 3 clusters


# SILHOUETTE SCORE METHOD
for num_clusters in range_n_clusters:
    # run the clustering
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    cluster_labels = kmeans.labels_

    # calculate the silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print(
        f'For {num_clusters} clusters, the average silhouette score is {silhouette_avg}')
