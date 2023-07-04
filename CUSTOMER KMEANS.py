#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Data collection and Analysis
cust=pd.read_csv(r"C:\Users\dines\OneDrive\Documents\CSV Files\Mall_Customers.csv")
print(cust.head(5))
"""print(cust.shape)
print(cust.info())
print(cust.describe())"""

#Choosing annual income and spending score columns
X=cust.iloc[:,[3,4]].values
print(X)

#Choosing the number of Clusters
#We can choose the right number of Clusters using WCSS
#WCSS - Within clusters sum of squares (Inertia)

#Finding WCSS (inertia) values for different no of clusters
"""wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
print(wcss)

#Plot an elblow graph to find which cluster has min value

sns.set() #This will give the basic themes and parameters
plt.plot(range(1,11),wcss)
plt.title("Elbow point graph")
plt.xlabel("No of clusters")
plt.ylabel("Inertia")
plt.show()"""

#Optimum no of clusters is 5

kmeans=KMeans(n_clusters=5)

#Return a label for each datapoin
Y=kmeans.fit_predict(X)
print(Y)

#Visualizing Clusters and centroids
plt.figure(figsize=(8,8)) #Giving x and y axis units
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c="green",label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c="cyan",label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c="yellow",label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c="blue",label='Cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c="black",label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,color="red",label="Centroid")
plt.title("Customer Clusters")
plt.xlabel("Annual income")
plt.ylabel("Spending scores")
plt.show()
