import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

class ClusterModel:
    def __init__(self):
        self.circle_data = None
        self.blob_data = None
        self.moon_data = None
        self.iris_data = None

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
            self.circle_data = self.mysql_to_df("select * from circle_data")
            self.blob_data = self.mysql_to_df("select * from blob_data")
            self.moon_data = self.mysql_to_df("select * from moon_data")
            self.iris_data = self.mysql_to_df("select * from iris")
            #转化为numpy数组
            self.circle_data = np.array(self.circle_data)
            self.blob_data = np.array(self.blob_data)
            self.moon_data = np.array(self.moon_data)
            #pca降维
            pca = PCA(n_components=2)
            self.iris_data_pca = pca.fit_transform(self.iris_data.iloc[:,1:-1])

    def k_clustering(self,k):
        kmeans = KMeans(n_clusters=k,n_init='auto')
        circle_labels = kmeans.fit_predict(self.circle_data)
        blob_labels = kmeans.fit_predict(self.blob_data)
        moon_labels = kmeans.fit_predict(self.moon_data)
        iris_labels = kmeans.fit_predict(self.iris_data_pca)
        return circle_labels,blob_labels,moon_labels,iris_labels
    def d_clustering(self,eps,min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        circle_labels = dbscan.fit_predict(self.circle_data)
        blob_labels = dbscan.fit_predict(self.blob_data)
        moon_labels = dbscan.fit_predict(self.moon_data)
        iris_labels = dbscan.fit_predict(self.iris_data_pca)
        return circle_labels,blob_labels,moon_labels,iris_labels


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
    label = clustering.dbscan_clustering(clustering.circle_data)
    clustering.plot_clusters(clustering.circle_data,label)