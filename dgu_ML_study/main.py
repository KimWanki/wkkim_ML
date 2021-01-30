
'''
author : wkkim
date : 2021.01.30
'''

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split




path = "/Users/kimwanki/Downloads/5강(1월 31일)/archive/X.npy"
answer_path = "/Users/kimwanki/Downloads/5강(1월 31일)/archive/Y.npy"

import numpy as np
X = np.load(path)
# y=np.load('Y.npy')
print(X.shape)

# X = X[0]
# print(X)
# print(X.shape)
#
# print(X[:,:,0].shape)
# print(X[:,:,1])
# print(X[:,:,2])
#
# print(X[:,:,0] + X[:,:,1])
#
# num = X[:,:,0] + X[:,:,1] +X[:,:,2]
#
# print(num.shape)
# print(num)
# result :
# 5547*50*50*3
# 226 + 164 + 206

arr2d = np.array(  [[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]],
                    [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]],
                    [[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]],
                    [[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]])

print(arr2d[:,:,0] + arr2d[:,:,1] + arr2d[:,:,2])
print(arr2d[:,:,0])
print(arr2d[:,:,1])
print(arr2d[:,:,2])


Y = np.load(answer_path)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print(x_train.shape)
print(y_train.shape)