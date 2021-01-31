# Programmer: HYUN WOOK KANG
# Date: 2021-Jan-23
# Description: 지난주 코드랑 어디가 바꼇을까? 정확도가 올라갔나?

from imutils import paths
from sklearn.preprocessing import OneHotEncoder
from models.networks import *
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skimage.exposure import equalize_hist

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--input_directory', required=True, default='trainig_set')
args = vars(parser.parse_args())

# imgPaths = list(paths.list_images(args['input_directory']))

size = 50

# X=[]
# y=[]
#
# for path in imgPaths:
#    img = cv2.imread(path)
#    img = cv2.resize(img, (size, size))
#    X.append(img)
#    label = path.split(os.path.sep)[-2]
#    y.append(label)
#
# X=np.array(X)
#
# X=0.2126*X[:,:,:,0]+0.7152*X[:,:,:,1]+0.0722*X[:,:,:,2]

# X=X/255.0
# why normalize?
# X=equalize_hist(X) #equalize_hist, optional

y = np.array(y).reshape(-1, 1)
X = X.reshape(X.shape + (1,))

oe = OneHotEncoder()
oe.fit(y)
y = oe.transform(y).toarray()

print(X.shape)
print(y.shape)

cnn = CNN(size)

model = cnn.build()

myadam = Adam(learning_rate=0.001)  # learning rate 란?
model.compile(optimizer=myadam, loss='categorical_crossentropy', metrics=['accuracy'])  # loss 란?

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=1)

model.fit(trainX, trainY, epochs=10, batch_size=5, verbose=0)
# returns the accuracy of the model
_, acc = model.evaluate(testX, testY)

print('accuracy: %.2f' % (100 * acc))

print(testY)

predictions = model.predict(testX)
print('predictions: ')
fw = open('result.txt', 'w')
for i in range(0, len(predictions)):
    fw.write(str(predictions[i][0]) + '\t' + str(predictions[i][1]) + '\n')

fw.close()




