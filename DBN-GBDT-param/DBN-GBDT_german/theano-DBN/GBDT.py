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
  

def classify(X, y):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    model.fit(X, y)
    return model
    
  
def main():
    X=[] 
    y=[]

    X = np.loadtxt("german2.csv",  delimiter = "," , usecols=(range(24)) , dtype=float)
    
    y = np.loadtxt("german2.csv",  delimiter = "," , usecols=(24,) , dtype=int)
 
    clf = classify(X, y)
 
    print m.classification_report(y, clf.predict(X))

 

if __name__ == "__main__":
    main()
