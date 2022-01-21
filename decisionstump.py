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

class DecisionStump:
    def __init__(self):
        self.threshold = None
        self.id_feature = None
        self.polarity = 1
        self.w_i = None

    def predict(self, X):
        m = X.shape[0]
        X_column = X[:, self.id_feature]
        predictions = np.ones(m)
        #cambia > o < in base alla polarity!
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions