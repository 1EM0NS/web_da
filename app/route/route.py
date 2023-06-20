import pymysql
from flask import Flask, request, jsonify
from app.models.Classification import ClassifyModel
from app.models.Cluster import ClusterModel
from app.models.Apriori import AprioriModel
from app.models.Regression import RegressionModel
import pandas as pd
import base64

app = Flask(__name__)


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    # mysql转df
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

    if request.method == 'POST':
        # 获取客户端传递的数据
        # noinspection PyTypeChecker
        data = request.get_json('data')
        height = data['height']
        leaf_samples = data['leaf_samples']
        # 调用分类分析模型进行处理
        model = ClassifyModel(height, leaf_samples)
        # 获取数据
        sql = "select * from iris"
        df = mysql_to_df(sql)
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
        # 发送转换为列表的数据
        return jsonify(df.values.tolist())


@app.route('/cluster', methods=['POST'])
def cluster():
    # noinspection PyTypeChecker
    k = request.get_json('k')["k"]
    # 调用聚类分析模型进行处理
    model = ClusterModel(k)
    label, data = model.cluster()
    result = {
        'label': label.tolist(),
        'data': data.tolist()
    }

    # 将处理结果返回给客户端
    return jsonify(result)


@app.route('/aprior', methods=['POST'])
def apriori():
    # 获取客户端传递的数据
    # noinspection PyTypeChecker
    data = request.get_json('data')
    min_support = data['min_support']
    min_confidence = data['min_confidence']

    # 调用关联分析模型进行处理
    model = AprioriModel(min_support, min_confidence)
    x, y, data = model.apriori_demo()
    result = {
        'x': x,
        'y': y,
        'data': data
    }
    # 将处理结果返回给客户端
    return jsonify(result)


@app.route('/regression', methods=['POST'])
def regression():
    # 获取客户端传递的数据
    # noinspection PyTypeChecker
    data = request.get_json('data')
    n_samples = data['n_samples']
    degree = data['degree']
    # 调用回归分析模型进行处理
    model = RegressionModel(degree)
    X, y = model.generate_data(n_samples)
    model.train(X, y)
    w, b = model.get_coefficients()
    result = {
        'x': X.tolist(),
        'y': y.tolist(),
        'w': w[1:].tolist(),
        'b': b.tolist()



    }
    # 将处理结果返回给客户端
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
