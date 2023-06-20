from sklearn.cluster import KMeans
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
        self.model = KMeans(n_clusters=k, random_state=42)

    def cluster(self,data):
        # 将数据转换为NumPy数组
        X = np.array(data[['SepL', 'SepW', 'PetL', 'PetW']])

        # 使用KMeans算法进行聚类分析
        self.model.fit(X)

        # 返回聚类结果
        return self.model.labels_


if __name__=='__main__':
    #导入数据库
    cnx = pymysql.connect(host='localhost',
                          user='root',
                          password='1784',
                          database='data_')
    cursor = cnx.cursor()
    sql = "select * from iris"
    cursor.execute(sql)
    data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    # Create a DataFrame with the data and column names
    df = pd.DataFrame(data, columns=column_names)
    c = ClusterModel()
    c.cluster(df)
    cnx.close()
