import numpy as np

class LogisticRegression:
    
    def __init__(self, group, learning_rate = 0.01, n_iters = 10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.group = group
        
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        y_ = [1 if i == self.group else 0 for i in y]
        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights)
            y_predict = self.sigmoid(linear_output)
            update = np.dot(X.T,(y_predict - y_))
            self.weights = self.weights - self.learning_rate * (update)
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        y_prob = self.sigmoid(linear_output)
        y_predict = np.where(y_prob > 0.5, 1, 0)
        return y_predict
    
    def predict_prob(self,X):
        linear_output = np.dot(X, self.weights)
        y_prob = self.sigmoid(linear_output)
        return y_prob
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
        

