import numpy as np
import cv2
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

img = 'C:/Users/USER/Desktop/wkkim/segmentation/total/Adeno/00142212 Adenocarcinoma0003.jpg'
data = 'C:/Users/USER/Desktop/wkkim/segmentation/total/Adeno/00142212 Adenocarcinoma0003.npy'

# input img,mask path
def mkRec(img, mask,size):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.load(mask).astype(np.uint8)
    h, w = image.shape[0], image.shape[1]

    # TODO : input size를 정사각형 형태로 맞춰주기.
    import math
    # # TODO : 가로로 붙이기.
    if h > w :
        image_bg = np.zeros(shape=(h, math.ceil((h - w) / 2), 3), dtype=np.uint8)
        mask_bg = np.zeros(shape=(h, math.ceil((h - w) / 2)), dtype=np.uint8)
        image = cv2.hconcat([image_bg, image, image_bg])
        mask = cv2.hconcat([mask_bg, mask, mask_bg])
    # TODO : 세로로 붙이기.
    elif h < w :
        image_bg = np.zeros(shape=(math.ceil((w-h)/2), w, 3), dtype=np.uint8)
        mask_bg = np.zeros(shape=(math.ceil((w-h)/2), w), dtype=np.uint8)
        image = cv2.vconcat([image_bg, image, image_bg])
        mask = cv2.vconcat([mask_bg, mask, mask_bg])

    image = cv2.resize(image, (size, size))
    mask = cv2.resize(mask, (size, size))

    return (image, mask)


image, mask = mkRec(img, data)

plt.imshow(mask)
plt.show()
plt.imshow(image)
plt.show()