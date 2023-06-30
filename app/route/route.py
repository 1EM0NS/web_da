import numpy as np
import pymysql
from flask import Flask, request, jsonify
from app.models.Classification import ClassifyModel
from app.models.Cluster import ClusterModel
from app.models.Apriori import AprioriModel
from app.models.Regression import RegressionModel
import pandas as pd
import base64
from flask_cors import CORS
from flasgger import Swagger
import os


def mysql_to_df(sq_):
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
os.environ["OMP_NUM_THREADS"] = "1"
app = Flask(__name__)
Swagger(app)

CORS(app, resources=r'/*', supports_credentials=True)
@app.route('/classify', methods=['POST', 'GET'])
def classify():

    # mysql转df
    if request.method == 'POST':

        # 获取客户端传递的数据
        # noinspection PyTypeChecker
        data = request.get_json('data')
        if 'height' not in data or 'leaf_samples' not in data:
            return jsonify({'code': 400, 'msg': 'Invalid request body'})
        height = data['height']
        leaf_samples = data['leaf_samples']
        # 调用分类分析模型进行处理
        model = ClassifyModel(height, leaf_samples)
        # 获取数据
        df = mysql_to_df("select * from iris")
        # 结果
        result = model.classify(df)
        # base64编码
        decision_tree_path = '../results/decision_tree.png'
        result_path = '../results/result.png'
        # 读取图片
        with open(decision_tree_path, 'rb') as f:
            decision_tree = base64.b64encode(f.read())
        with open(result_path, 'rb') as f:
            result_img = base64.b64encode(f.read())
        # 将处理结果返回给客户端
        result = {
            'decision_tree': str(decision_tree)[2:-1],
            'result_img': str(result_img)[2:-1],
            'result': result.tolist()
        }
        # 将处理结果返回给客户端
        return jsonify(result)
    else:
        # 获取数据库数据
        sql = "select * from iris"
        df = mysql_to_df(sql)
        #df数量
        df_num = len(df)
        result = {
            'total': df_num,
            'df': df.values.tolist()
        }
        # 发送转换为列表的数据
        return jsonify(result)


@app.route('/cluster', methods=['POST', 'GET'])
def cluster():
    model = ClusterModel()
    model.generate_data()
    if request.method == 'POST':
        # 获取客户端传递的数据
        data = request.get_json('data')
        # 判断数据是否为空
        if not data['model']:
            return jsonify({'error': 'model is not defined'})
        mode = data['model']
        # 调用分类分析模型进行处理
        if mode == 'kmeans':
            if not data['k']:
                return jsonify({'error': 'k is not defined'})
            k = data['k']
            circle_label,blob_label,moon_label,iris_label = model.k_clustering(k)

        if mode == 'dbscan':
            if not data['min_samples'] or not data['eps']:
                return jsonify({'error': 'min_samples or eps is not defined'})
            min_samples = data['min_samples']
            eps = data['eps']
            circle_label,blob_label,moon_label,iris_label = model.d_clustering(eps, min_samples)
            # model.plot_clusters(model.circle_data, circle_label)
            # model.plot_clusters(model.blob_data, blob_label)
            # model.plot_clusters(model.moon_data, moon_label)
            # model.plot_clusters(model.iris_data_pca, iris_label)
        blob = [{'data': model.blob_data[i].tolist(), 'label': int(blob_label[i])} for i in range(len(blob_label))]
        circle = [{'data': model.circle_data[i].tolist(), 'label': int(circle_label[i])} for i in range(len(circle_label))]
        moon = [{'data': model.moon_data[i].tolist(), 'label': int(moon_label[i])} for i in range(len(moon_label))]
        iris = [{'data': model.iris_data_pca[i].tolist(), 'label': int(iris_label[i])} for i in range(len(iris_label))]

        result = {
            'circle': circle,
            'blob': blob,
            'moon': moon,
            'iris': iris
        }
        return jsonify(result)
    else:
        result = {
            'circle_data': model.circle_data.tolist(),
            'blob_data': model.blob_data.tolist(),
            'moon_data': model.moon_data.tolist(),
            'iris_data': model.iris_data.values.tolist(),
        }
        return jsonify(result)

@app.route('/aprior', methods=['POST','GET'])
def apriori():

    if request.method== 'POST':
        # 获取客户端传递的数据
        data = request.get_json('data')
        min_support = data['min_support']
        min_confidence = data['min_confidence']
        # 判断是否有缺失值
        if any([min_support is None, min_confidence is None]):
            min_support = 0.3
            min_confidence = 0.3
        # 调用关联分析模型进行处理
        model = AprioriModel(min_support, min_confidence)
        list1,list2,list3 = model.serborn_get()
        #两位小数
        list2 = np.round(list2, 2).tolist()
        list3 = np.round(list3, 2).tolist()
        result = {
            'label': list1,
            'confidence': list2,
            'support': list3,
        }
        # 将处理结果返回给客户端
        return jsonify(result)
    else:
        model = AprioriModel(0.3, 0.3)
        data = model.loadDataSet()
        data_table = [{'item{}'.format(j):data[i][j] for j in range(len(data[i]))} for i in range(len(data))]
        result = {
            'data': data,
            'data_table':data_table
        }
        return jsonify(result)

@app.route('/regression', methods=['POST'])
def regression():
    # 获取客户端传递的数据
    data = request.get_json('data')
    # 检查数据是否有效
    if not data['n_samples']:
        return jsonify({'error': 'n_samples is required'})
    n_samples = data['n_samples']
    if not data['degree']:
        return jsonify({'error': 'degree is required'})
    degree = data['degree']
    # 调用回归分析模型进行处理
    model = RegressionModel(degree)
    X, y = model.generate_data(n_samples)
    model.train(X, y)
    w, b = model.get_coefficients()
    #x,y合并
    X = np.column_stack((X, y))
    result = {
        'data': X.tolist(),
        'w': w[1:].tolist(),
        'b': b.tolist()
    }
    # 将处理结果返回给客户端
    return jsonify(result)
@app.route('/', methods=['GET'])
def index():
    return 'hello world'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
