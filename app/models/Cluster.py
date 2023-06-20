from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pymysql
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE
class ClusterModel:
    def __init__(self,k=3):
        self.k = k
        self.model = KMeans(n_clusters=k)
        # 生成聚类散点数据
        self.data,self.labels = make_blobs(n_samples=100, n_features=2, centers=k, cluster_std=1.5, shuffle=True, random_state=42)

    def cluster(self):

        # 使用KMeans算法进行聚类分析
        self.model.fit(self.data)

        # 返回聚类结果
        return self.model.labels_,self.data


if __name__=='__main__':
    model = ClusterModel()
    labels,data = model.cluster()
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.show()
