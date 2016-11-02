#!/usr/bin/python3

import numpy as np
from sklearn.linear_model import ARDRegression

N = 30

X = np.random.random(size=(N,N)) * 10 + 1

w = np.zeros(N)
w[:5] = np.arange(5) + 1

e = np.random.normal(0, 1, size=N)

Y = np.dot(X, w) + e

ard = ARDRegression()
ard.fit(X, Y)

coef = list(map(lambda x: round(x, 2), ard.coef_))
print(coef)

