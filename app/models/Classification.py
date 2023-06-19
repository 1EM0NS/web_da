import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pymysql
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE

class ClassifyModel:
    def __init__(self, height=3, leaf_samples=2):
        self.height = height
        self.leaf_samples = leaf_samples
        self.model = DecisionTreeClassifier(max_depth=height, min_samples_leaf=leaf_samples)

    def classify(self, data):
        # 取每个元组的前四个
        column_names = ['SepL', 'SepW', 'PetL', 'PetW']
        X = np.array(data[column_names])
        y = np.array(data['Species'])

        # 使用DecisionTreeClassifier算法进行分类分析
        self.model.fit(X, y)

        # 绘制决策树
        plt.figure(figsize=(20,20))
        plot_tree(self.model, filled=True,class_names=['setosa','versicolor','virginica'],feature_names=column_names)
        plt.savefig('../results/decision_tree.png')

        # 绘制分类结果
        plt.figure(figsize=(20,10))
        encoder = LE()
        numerical_categories = encoder.fit_transform(self.model.predict(X))
        plt.scatter(X[:, 0], X[:, 1], c=numerical_categories)
        plt.savefig('../results/result.png')

        # 返回分类结果
        return self.model.predict(X)


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
    c = ClassifyModel()
    c.classify(df)
    cnx.close()
