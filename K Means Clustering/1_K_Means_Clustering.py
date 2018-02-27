# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal # of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    #max iter= maximum # of iterations to find cluster
    #n init= number of times kmeans will be run with different initial centroids
    kmeans = KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#The previous code showed 5 clusters being optimal so now we run
#almost similiar code but with n_clusters=5
#Applying K Means to the mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the clusters
#To visulize the zeros cluster from y_kmeans, put y_kmeans ==0 from the first column which is zero
#then y_kmeans==0 and 1 for y cordinate
#The rest are shown with the others clusters and the number are changed respective to their cluster number
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s=100,c='red',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s=100,c='cyan',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s=100,c='magenta',label='Sensible')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
