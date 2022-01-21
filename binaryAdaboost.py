import pandas as pd
import numpy as np
import math
import seaborn as sns
from math import exp
import matplotlib.pyplot as plt
import random

from decisionstump import DecisionStump

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import Counter

class Adaboost:
    def __init__(self, n_stumps=5):
        self.n_stumps = n_stumps
        self.clfs = []

    def fit(self, X, y, rand = False, verbose=False):
        m, d = X.shape
        pesi = np.full(m, (1 / m)) #pesi tutti 1/m alla prima iterazione

        self.clfs = []
        self.ls_w_i = []
        for _ in range(self.n_stumps):
            clf = DecisionStump()
            min_error = np.inf
            
            if rand == True:
                max_iteration=0
                while min_error > 0.47 and max_iteration < 50:
                    max_iteration+=1
                    feature_i = random.randint(0,d-1)
                    column = X[:, feature_i]
                    thresholds = np.unique(column)
                    threshold = random.choice(thresholds)
                        
                    p = 1
                    predictions = np.ones(m)
                    predictions[column < threshold] = -1
                    errati = pesi[y != predictions]
                    error = sum(errati)
                    if error > 0.53:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.threshold = threshold
                        clf.id_feature = feature_i
                        clf.polarity = p
                        min_error = error
                    if verbose == True: 
                        print(max_iteration,'  threshold =', threshold, '      id_feat = ',feature_i, '    error = ', round(min_error, 3))
            if rand == False:
                for feature_i in range(d):
                    column = X[:, feature_i]
                    thresholds = np.unique(column)
                    for threshold in thresholds:
                        p = 1
                        predictions = np.ones(m)
                        predictions[column < threshold] = -1
                        misclassified = pesi[y != predictions]
                        error = sum(misclassified)
                        if error > 0.5:
                            error = 1 - error
                            p = -1
                        if error < min_error:
                            clf.threshold = threshold
                            clf.id_feature = feature_i
                            clf.polarity = p
                            min_error = error
                        
            if round(min_error, 3) == 0: 
                min_error += 0.001
                
            try:
                clf.w_i = 0.5 * np.log((1.0 - min_error) / (min_error))
                self.ls_w_i.append(clf.w_i)
            except:
                print('break')
                break
            
            predictions = clf.predict(X)
            
            
            pesi = np.multiply(pesi, np.exp(-clf.w_i * y * predictions))
            pesi = pesi/np.sum(pesi)
            
            self.clfs.append(clf)
        return self.clfs, self.ls_w_i

    def predict(self, X, clfs = None, ls_w_i= None, weighted=False):
        if clfs is None:
            clf_preds = [clf.w_i * clf.predict(X) for clf in self.clfs]
        else:
            clf_preds = [ ls_w_i[index]* clf.predict(X) for index, clf in enumerate(clfs)]
            
        if weighted == False: 
            y_pred = np.sum(clf_preds, axis=0)
            y_pred = np.sign(y_pred)
            return y_pred
        else: 
            y_pred = np.sum(clf_preds, axis=0)
            return y_pred