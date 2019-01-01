# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.drop(['User ID','Purchased'],axis = 1)
    y = dataset['Purchased'].values
    y = y.reshape(-1,1)

    X = pd.get_dummies(X, columns = ['Gender'])
    X = X.drop(['Gender_Male'],axis = 1)
    
    
    X['Age'] = (X['Age'] - X['Age'].mean()) / X['Age'].std()
    X['EstimatedSalary'] = (X['EstimatedSalary'] - X['EstimatedSalary'].mean()) / X['EstimatedSalary'].std()

    X = X.values
    w = np.random.randn(X.shape[1])
    y_pred,w = training(X,y,w)
    return y_pred,w

learning_rate = 0.003
J = []


def compute_cost(y_pred,y):
    a = ((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))) * (-1 / len(y))
    return a.sum()

def derivative(y_pred,y,X):
    b = sum((y_pred - y) * X)
    return b

def training(X,y,w):
    for i in range(10000):
        y_pred = (1 / (1 + np.exp(-(X * w).sum(axis=1,keepdims=True))))
        cost = compute_cost(y_pred,y)
        print('Cost Function::'+str(cost))
        w = w - (learning_rate / len(y) *(derivative(y_pred,y,X)))
        J.append(cost)    
    return y_pred,w
       

if __name__ == '__main__':
    y_pred,w = main()  
    plt.plot(J)      
