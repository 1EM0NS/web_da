import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

class ClusterModel:
    def __init__(self):
        self.data = None
        self.labels = None

    def mysql_to_df(self,sq_):
        cnx = pymysql.connect(host='localhost',
                              user='root',
                              password='1784',
                              database='data_')
        cursor = cnx.cursor()
        cursor.execute(sq_)
        data_ = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        # Create a DataFrame with the data and column names
        df_ = pd.DataFrame(data_, columns=column_names)
        cnx.close()
        return df_

    def generate_data(self):
            self.circle_data, _ = make_circles(n_samples=100, noise=0.05, factor=0.5)
            self.blob_data, _ = make_blobs(n_samples=100, centers=3)
            self.moon_data, _ = make_moons(n_samples=100, noise=0.05)
            df_iris = self.mysql_to_df("select * from iris")
            #pca降维
            pca = PCA(n_components=2)
            self.iris_data = pca.fit_transform(df_iris.iloc[:,1:-1])
            return self.circle_data, self.blob_data, self.moon_data

    def kmeans_clustering(self,data, n_clusters=3):
        kmeans = KMeans(n_clusters=3,n_init='auto')
        labels = kmeans.fit_predict(data)
        return labels

    def dbscan_clustering(self,data,min_samples=20, eps=0.5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        return labels

    def plot_clusters(self,data,labels):
        unique_labels = np.unique(labels)
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'orange', 'purple', 'brown', 'pink']

        for label in unique_labels:
            cluster_points = data[labels == label]
            plt.scatter(cluster_points[:,0], cluster_points[:,1], c=colors[label], label=f'Cluster {label + 1}')

        plt.legend()
        plt.show()

if __name__ == '__main__':
    clustering = ClusterModel()
    clustering.generate_data()
    label = clustering.dbscan_clustering(clustering.iris_data)
    clustering.plot_clusters(clustering.iris_data,label)