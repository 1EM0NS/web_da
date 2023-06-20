import pymysql
from flask import Flask, request, jsonify,send_file
from app.models.Classification import ClassifyModel
from app.models.Cluster import ClusterModel
from app.models.Apriori import AprioriModel
import pandas as pd
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    # 获取客户端传递的数据
    height = request.json['height']
    leaf_samples = request.json['leaf_samples']

    # 调用分类分析模型进行处理
    model = ClassifyModel(height, leaf_samples)

    #获取数据
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

    #结果
    result = model.classify(pd)
    decision_tree_path = './results/decision_tree.png'
    result_path = './results/result.png'

    # 将处理结果返回给客户端
    return jsonify(result),send_file(decision_tree_path), send_file(result_path)


@app.route('/cluster', methods=['POST'])
def cluster():
    # 获取客户端传递的数据
    data = request.json['data']
    k = request.json['k']

    # 调用聚类分析模型进行处理
    model = ClusterModel(k)
    result = model.cluster(data)

    # 将处理结果返回给客户端
    return jsonify(result)

@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello World!'

@app.route('/aprior', methods=['POST'])
def apriori():
    # 获取客户端传递的数据
    data = request.json['data']
    min_support = request.json['min_support']
    min_confidence = request.json['min_confidence']

    # 调用关联分析模型进行处理
    model = AprioriModel(min_support, min_confidence)
    result = model.apriori(data)

    # 将处理结果返回给客户端
    return jsonify(result)

if __name__ == '__main__':
    app.run()