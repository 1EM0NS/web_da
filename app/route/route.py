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
os.environ["OMP_NUM_THREADS"] = "4"
app = Flask(__name__)
Swagger(app)

CORS(app, resources=r'/*', supports_credentials=True)
@app.route('/classify', methods=['POST', 'GET'])
def classify():
    """
        分类接口
        ---
        tags:
            - 分类接口
        post:
          summary: Perform classification analysis on iris dataset
          description: |
            This endpoint performs classification analysis on the iris dataset using the provided parameters and returns the decision tree, result image, and classification results.
          parameters:
            - name: body
              in: body
              required: true
              schema:
                id: Classify
                required:
                    - height
                    - leaf_samples
                properties:
                    height:
                        type: integer
                        description: The height of the decision tree
                    leaf_samples:
                        type: integer
                        description: The minimum number of samples required to be at a leaf node
        definitions:
          Classify:
            type: object
            properties:
              height:
                type: integer
                description: The height of the decision tree
              leaf_samples:
                type: integer
                description: The minimum number of samples required to be at a leaf node
        responses:
            200:
              description: Successfully performed classification analysis
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      decision_tree:
                        type: string
                        description: The base64-encoded PNG image of the decision tree
                      result_img:
                        type: string
                        description: The base64-encoded PNG image of the classification result
                      result:
                        type: array
                        description: The classification result
                        items:
                          type: number
                  example:
                    decision_tree: "iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAYAAAB5fY51AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABZ0RVh0Q3JlYXRpb24gVGltZQAxNjEwNzYwNzYyNTYy.png"
                    result_img: "iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAYAAAB5fY51AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABZ0RVh0Q3JlYXRpb24gVGltZQAxNjEwNzYwNzYyNTYy.png"
                    result: [0, 1, 2, 0, 1, 2, ...]
            400:
              description: Invalid request body
            500:
              description: Error occurred during classification analysis
        get:
          summary: Get iris dataset
          description: |
            This endpoint retrieves the iris dataset from the database and returns it as a list of lists.
        responses:
            200:
              description: Successfully retrieved iris dataset
              content:
                application/json:
                  schema:
                    type: array
                    items:
                      type: array
                      items:
                        type: string
                  example:
                    [["5.1", "3.5", "1.4", "0.2", "Iris-setosa"], ["4.9", "3.0", "1.4", "0.2", "Iris-setosa"], ...]
            500:
              description: Error occurred while retrieving data from the database
        """
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
        data = request.get_json('data')
        mode = data['model']
        if mode == 'kmeans':
            k = data['k']
            circle_label = model.kmeans_clustering(model.circle_data,k)
            blob_label = model.kmeans_clustering(model.blob_data,k)
            moon_label = model.kmeans_clustering(model.moon_data,k)
            result = {
                'circle_label': circle_label.tolist(),
                'blob_label': blob_label.tolist(),
                'moon_label': moon_label.tolist(),
                'circle_data': model.circle_data.tolist(),
                'blob_data': model.blob_data.tolist(),
                'moon_data': model.moon_data.tolist()
            }
        if mode == 'dbscan':
            min_samples = data['min_samples']
            eps = data['eps']
            circle_label = model.dbscan_clustering(model.circle_data,min_samples,eps)
            blob_label = model.dbscan_clustering(model.blob_data,min_samples,eps)
            moon_label = model.dbscan_clustering(model.moon_data,min_samples,eps)
            result = {
                'circle_label': circle_label.tolist(),
                'blob_label': blob_label.tolist(),
                'moon_label': moon_label.tolist(),
                'circle_data': model.circle_data.tolist(),
                'blob_data': model.blob_data.tolist(),
                'moon_data': model.moon_data.tolist()
            }
        return jsonify(result)
    else:
        result = {
            'circle_data': model.circle_data.tolist(),
            'blob_data': model.blob_data.tolist(),
            'moon_data': model.moon_data.tolist()
        }
        return jsonify(result)



@app.route('/aprior', methods=['POST','GET'])
def apriori():
    """
        ---
        tags:
            - 关联分析接口
        post:
          summary: Perform association analysis on transaction data
          description:
            关联分析接口，json格式
          parameters:
            - name: body
              in: body
              required: true
              schema:
                id: 关联分析参数
                required:
                    - min_support
                    - min_confidence
                properties:
                    min_support:
                        type: integer
                        description: The minimum support threshold for the association rules
                    min_confidence:
                        type: integer
                        description: The minimum confidence threshold for the association rules
        definitions:
          关联分析参数:
            type: object
            properties:
              min_support:
                type: number
                description: The minimum support threshold
                default: 0.5
              min_confidence:
                type: number
                description: The minimum confidence threshold
                default: 0.5
        responses:
            200:
              description: Successfully performed association analysis
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      x:
                        type: array
                        description: The support values of the association rules
                        items:
                          type: number
                      y:
                        type: array
                        description: The confidence values of the association rules
                        items:
                          type: number
                      data:
                        type: array
                        description: The lift values of the association rules
                        items:
                          type: number
                  example:
                    x: [0.1, 0.2, 0.3, ...]
                    y: [0.5, 0.6, 0.7, ...]
                    data: [1.0, 1.2, 1.4, ...]
            400:
              description: Invalid request body
            500:
              description: Error occurred during association analysis
        get:
          summary: Get transaction data
          description: |
            This endpoint retrieves the transaction data from the database and returns it as a list of lists.
          responses:
            '200':
              description: Successfully retrieved transaction data
              content:
                application/json:
                  schema:
                    type: array
                    items:
                      type: array
                      items:
                        type: string
                  example:
                    [["apple", "banana", "orange"], ["apple", "banana"], ...]
            '500':
              description: Error occurred while retrieving data from the database
        """
    if request.method== 'POST':
        # 获取客户端传递的数据
        data = request.get_json('data')
        min_support = data['min_support']
        min_confidence = data['min_confidence']
        if any([min_support is None, min_confidence is None]):
            min_support = 0.3
            min_confidence = 0.3
        # 调用关联分析模型进行处理
        model = AprioriModel(min_support, min_confidence)
        list1,list2,list3 = model.serborn_get()
        #两位小数
        list2 = np.round(list2, 2).tolist()
        list3 = np.round(list3, 2).tolist()


        #list转置
        # list2 = list(map(list, zip(*list2)))
        # list3 = list(map(list, zip(*list3)))


        result = {
            'label': list1,
            'confidence': list2,
            'support': list3
        }
        # 将处理结果返回给客户端
        return jsonify(result)
    else:
        model = AprioriModel(0.3, 0.3)
        result = {
            'data': model.loadDataSet()
        }
        return jsonify(result)

@app.route('/regression', methods=['POST'])
def regression():
    """
        ---
        tags:
            - 回归分析接口
        post:
          summary: Perform regression analysis on generated data
          description: |
            This endpoint generates data based on the provided parameters, trains a regression model on the generated data, and returns the coefficients of the trained model.
          parameters:
            - name: body
              in: body
              required: true
              schema:
                id: 回归分析参数
                required:
                    - n_samples
                    - degree
                properties:
                    n_samples:
                        type: integer
                        description: The number of data samples to generate
                    degree:
                        type: integer
                        description: The degree of the polynomial to fit to the generated data
        definitions:
          回归分析参数:
            type: object
            properties:
              n_samples:
                type: integer
                description: The number of data samples to generate
                default: 100
              degree:
                type: integer
                description: The degree of the polynomial to fit to the generated data
                default: 2
          requestBody:
            required: true
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    n_samples:
                      type: integer
                      description: The number of data samples to generate
                    degree:
                      type: integer
                      description: The degree of the polynomial to fit to the generated data
                  example:
                    n_samples: 100
                    degree: 2
        responses:
            200:
              description: Successfully performed regression analysis
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      x:
                        type: array
                        description: The generated x values
                        items:
                          type: number
                      y:
                        type: array
                        description: The generated y values
                        items:
                          type: number
                      w:
                        type: array
                        description: The coefficients of the trained model
                        items:
                          type: number
                      b:
                        type: number
                        description: The intercept of the trained model
                  example:
                    x: [0.0, 0.010101010101010102, 0.020202020202020204, ...]
                    y: [0.0, 0.010101010101010102, 0.020202020202020204, ...]
                    w: [0.0, 1.0000000000000002, -0.0]
                    b: 0.0
            400:
              description: Invalid request body
            500:
              description: Error occurred during regression analysis
        """
    # 获取客户端传递的数据
    # noinspection PyTypeChecker
    data = request.get_json('data')
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
