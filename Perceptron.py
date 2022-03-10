import numpy as np

class perceptron:
    def __init__(self, group, learning_rate = 0.01, n_iters = 10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.group = group
        self.activation_func = self.stable_sigmoid
        self.weights = None

    def stable_sigmoid(self,x): #not stable, just ignore warning

        sig = 1 / (1 + np.exp(-x))
    
        return np.where(sig>=0.5,1,0)

        
 

    def predict(self, X):
        linear_output = np.dot(X,self.weights)
        y_pred = self.activation_func(linear_output)
        #y_pred = np.where(y_pred>=0.5,1,0)

        return y_pred

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features) #guess a weight 
        y_ = [1 if i == self.group else 0 for i in y]
        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights)
                y_predict = self.activation_func(linear_output)
                update = self.learning_rate * (y_[idx] - y_predict)
                self.weights = self.weights + (update * x_i)



    

    