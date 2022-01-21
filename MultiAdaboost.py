import pandas as pd
import numpy as np
import math
import seaborn as sns
from math import exp
import matplotlib.pyplot as plt
import random


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import Counter



from binaryAdaboost import Adaboost
from decisionstump import DecisionStump

class MultiAdaboost:
    def __init__(self,n_clf, classi):
        self.n_clf = n_clf
        self.classi = classi
        self.classificatori_binari = []
        
    def fit(self, df, y, rand=False):
        for classe in self.classi:
            training_labels = y.copy()
            train_all_vs_one = np.array([1 if v == classe else -1 for v in training_labels ])
            print(Counter(train_all_vs_one))
            clf = Adaboost(n_stumps=self.n_clf)
            ada, ls_w_i = clf.fit(df, train_all_vs_one, rand=rand)
            self.classificatori_binari.append((ada,ls_w_i))
        return self.classificatori_binari

    def predict(self, ls_predizioni):
        return np.array([np.argmax(v)+1 for v in zip(*ls_predizioni)])
        