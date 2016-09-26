#!usr/bin/env python  
#-*- coding: utf-8 -*-  
#https://www.52ml.net/16044.html


import sys  
import os  
import time  
import numpy as np  
import cPickle as pickle  
import numpy as np
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as m

reload(sys)  
sys.setdefaultencoding('utf8')  
  
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(1000, noise=0.25)
    with open("samples.txt",'w') as FILEOUT1:
        for i in X:
            print>>FILEOUT1,i[0],"\t",i[1]
    with open("labels.txt",'w') as FILEOUT2:
        for j in y:
            print>>FILEOUT2,j

def visualize(X, y, clf):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print xx.shape 
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    print Z.shape
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def classify(X, y):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    #model = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, loss='ls')
    model.fit(X, y)
    return model
    
  
def main():
    generate_data()
    X=[] 
    y=[]
 
    X1 = pd.read_csv("samples.txt",sep='\t',header=None)   #pandas dataframe 
    X= X1.values  #numpy type
    #X = np.loadtxt("samples.txt",  delimiter = "\t" , usecols=(0,1) , dtype=float)  #direct numpy method
    y = np.loadtxt("labels.txt" , dtype=int)
    clf = classify(X, y)
    visualize(X, y, clf)

    precision, recall, thresholds = m.precision_recall_curve(y, clf.predict(X))
    ax = plt.subplot(2, 1, 2)
    ax.plot(recall, precision)
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision recall curve')
    plt.show()
    #print(m.classification_report(y, clf.predict(X)) )

    fpr, tpr, thresholds = m.roc_curve(y, clf.predict(X))
    roc_auc = m.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    


if __name__ == "__main__":
    main()
