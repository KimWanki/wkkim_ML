import numpy as np
import matplotlib.image as mpimg

img_npy = np.load('C:/Users/USER/Downloads/Keras-Class-Activation-Map-master/image/total/HGD/00048689 HGD.npy')
img_path = mpimg.imread('C:/Users/USER/Downloads/Keras-Class-Activation-Map-master/image/total/HGD/00048689 HGD.jpg')
from matplotlib import pyplot as plt

plt.imshow(img_npy, cmap='gray')
plt.show()
print(img_npy.size)
print(img_npy.shape)

print(img_path.size)
print(img_path.shape)

plt.imshow(img_path)
plt.show()
