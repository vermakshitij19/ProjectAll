# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:31:28 2025

@author: Kshitij
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Kshitij\Downloads\final1.csv")

X = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
knn_classifier= KNeighborsClassifier()

