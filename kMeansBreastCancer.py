"""
Import statements
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import urllib2
import csv

"""
Get data from url, convert it into csv
"""
response = urllib2.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
data_retrieved = csv.reader(response)

"""
Create a matrix
"""
data = []
for row in data_retrieved:
    data.append(row)
"""
Data cleaning: replace ? with -1
"""
for vector in data:
    for i in range(1,11):
        vector[i] = int(vector[i]) if vector[i] != "?" else -1

"""
Create a pandas dataframe
"""
X = np.array(data)

df = pd.DataFrame(X)

"""
Create feature arrays to use for clustering
"""

f1 = df[1].values
f2 = df[2].values
f3 = df[3].values
f4 = df[4].values
f5 = df[5].values
f6 = df[6].values
f7 = df[7].values
f8 = df[8].values
f9 = df[9].values

Y = np.matrix(zip(f1,f2,f3,f4,f5,f6,f7,f8,f9))

kmeans = KMeans(n_clusters=2).fit(Y)

print "Indexed, labelled data:", kmeans.labels_

print "Means of the two clusters:", kmeans.cluster_centers_


"""
Finally, find accuracy of the clustering algorithm
"""
correct = 0
manual_classification = df[10].values

for i in range(699):
    if (manual_classification[i] == '2' and kmeans.labels_[i] == 0) or (manual_classification[i] == '4' and kmeans.labels_[i] == 1):
        correct += 1

print "Accuracy:", correct * 100.0 / 699
