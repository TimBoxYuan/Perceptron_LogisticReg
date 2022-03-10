# implement Percertron 

from keras.datasets import mnist
import numpy as np
from Perceptron import perceptron
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


# load and organize data
### use only the first 20,000 training images
### only the first 2,000 testing images
X_train = []
for i in range(20000):
    array = list(train_X[i].flatten())
    array.insert(0,1) # add bias term
    X_train.append(array)
    
X_train = np.array(X_train)

y_train = np.array(train_y[0:20000])

X_test = []
for j in range(2000):
    array = list((test_X[j].flatten()))
    array.insert(0,1) # add the bias term x0 = 1
    X_test.append(array)
X_test = np.array(X_test)

y_test = np.array(test_y[0:2000])

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)

# Part 1 (find accuracy of class 0 vs class not 0, ... for all classes)
sig_list = []  
acc_list = []
for c in range(10):
    y_t0 = [1 if i == c else 0 for i in y_test]
    p = perceptron(c, learning_rate = 0.01, n_iters = 10)
    p.fit(X_train, y_train)
    predictions_0 = p.predict(X_test)
    sig0 = 1 / (1 + np.exp(-np.dot(X_test, p.weights)))

    sig_list.append(sig0)
    accuracy = np.sum(y_t0 == predictions_0)/len(y_test)
    acc_list.append(accuracy)

print('Perceptron accuracy list for all classes', acc_list)

# Part 2 (find the accuracy on how many test images have the highest probability
# at the right index)

df3 = pd.DataFrame(sig_list) 
#each row is a class run, each column is the probs of the image for 10 class runs

q3 = np.array(df3.idxmax(axis=0)) #column wise find the index of the max prob
accuracy = sum(q3 == y_test)/len(y_test)

print(' Perceptron Part 2 accuracy is', accuracy)




############ Logistic Regression
from LR import LogisticRegression

prob_list_LR = []  
acc_list_LR = []
for c in range(10):
    y_t0 = [1 if i == c else 0 for i in y_test]
    LR = LogisticRegression(c, learning_rate = 0.01, n_iters = 20) #changed from default
    LR.fit(X_train, y_train)
    
    predictions_0 = LR.predict(X_test)
    prob = LR.predict_prob(X_test)
    prob_list_LR.append(prob)
    
    accuracy = np.sum(y_t0 == predictions_0)/len(y_test)
    acc_list_LR.append(accuracy)

print('Logistic Regression accuracy list for all classes', acc_list_LR)

## Part 2 (find the accuracy on how many test images have the highest probability
# at the right index)
df4 = pd.DataFrame(prob_list_LR)
q4 = np.array(df4.idxmax(axis=0))
accuracy_LR = sum(q4 == y_test)/len(y_test)

print(' LR Part 2 accuracy is', accuracy_LR)