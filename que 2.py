import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
p=pd.read_csv("./K-Mean_Dataset.csv")
print(p.head())
Q = p.iloc[:,1:].values

R = SimpleImputer(missing_values=np.nan, strategy='mean')
R = R.fit(Q)
Q = R.transform(Q)

z = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(Q)
    z.append(kmeans.inertia_)
plt.plot(range(1,11),z)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Z')
plt.show()


nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
y=km.fit(Q)
print(y)


y_cluster_kmeans = km.predict(Q)
score = metrics.silhouette_score(Q, y_cluster_kmeans)
print('Silhouette score:',score)
scaler = preprocessing.StandardScaler()
scaler.fit(Q)
X_scaled_array = scaler.transform(Q)
X_scaled = pd.DataFrame(X_scaled_array)
nclusters = 4
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled)
y_scaled_cluster_kmeans = km.predict(X_scaled)
score = metrics.silhouette_score(X_scaled, y_scaled_cluster_kmeans)
print('Silhouette score after applying scaling:',score)
