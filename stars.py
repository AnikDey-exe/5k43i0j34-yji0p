import csv 
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv("stars.csv")
SizeList = df["Size"].to_list()
LightList = df["Light"].to_list()

X = df.iloc[:, [0, 1]].values
print(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init ='k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
# plt.show()

kmeans = KMeans(n_clusters = 3, init ='k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=100,marker=',')
plt.grid(False)
plt.title('Clusters of Light')
plt.xlabel('Size')
plt.ylabel('Light')
plt.legend()
plt.show()

# print(df.head())

# fig = px.scatter(x = SizeList, y = LightList, title = 'Sizes')
# fig.show()