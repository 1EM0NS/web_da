from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
class RegressionModel:
    def __init__(self, degree):
        self.degree = degree
        self.model = LinearRegression()
        self.X,self.y = None,None
    def generate_data(self, n_samples=100):
        # Generate random data for training
        #顺序数据
        X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                              n_informative=1, noise=10,
                                              coef=True, random_state=0)

        # Add outlier data
        np.random.seed(0)
        X[:100] = 3 + 0.5 * np.random.normal(size=(100, 1))
        X = X.squeeze()
        y[:100] = -3 + 10 * np.random.normal(size=100)
        self.X,self.y = X,y
        return self.X,self.y

    def train(self, X, y):
        # Transform the input data to include polynomial features up to the specified degree

        X_poly = np.polynomial.polynomial.polyvander(X, self.degree)
        self.model.fit(X_poly, y)

    def predict(self, X):
        # Transform the input data to include polynomial features up to the specified degree
        X_poly = np.polynomial.polynomial.polyvander(X, self.degree)
        return self.model.predict(X_poly)

    def get_coefficients(self):
        # Get the coefficients (w) and intercept (b) of the linear regression model
        return self.model.coef_, self.model.intercept_

if __name__=='__main__':
    r = RegressionModel(4)
    X,y = r.generate_data(n_samples=500)
    r.train(X,y)
    plt.scatter(X,y)
    #建立区间
    X_test = np.linspace(-2,2,100)
    plt.plot(X_test,r.predict(X_test),c='r',linewidth=3)
    print(r.predict(X))
    print(r.get_coefficients())
    plt.show()