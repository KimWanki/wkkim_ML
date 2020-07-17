import os
import random
import shutil
img_dir = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/dataset/train/'
npy_dir = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/dataset/trainannot/'


train_img = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/train/'

train_npy = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/trainannot/'

test_img = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/test/'
test_npy = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/testannot/'

val_img = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/val/'
val_npy = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/valannot/'

img_list = os.listdir(img_dir)

random.shuffle(img_list)
random.shuffle(img_list)

_train = img_list[:2100]
_val = img_list[2100:2700]
_test = img_list[2700:]
print(len(_train))
count = 0
for i in _train:
    img_label = i
    npy_label = i.replace('tif', 'bmp').replace('bmp', 'jpg').replace('jpg', 'npy')
    shutil.move(img_dir+img_label, train_img+img_label)
    shutil.move(npy_dir+npy_label, train_npy+npy_label)

for i in _val:
    img_label = i
    npy_label = i.replace('tif', 'bmp').replace('bmp', 'jpg').replace('jpg', 'npy')
    shutil.move(img_dir + img_label, val_img + img_label)
    shutil.move(npy_dir + npy_label, val_npy + npy_label)

for i in _test:
    img_label = i
    npy_label = i.replace('tif', 'bmp').replace('bmp', 'jpg').replace('jpg', 'npy')
    shutil.move(img_dir + img_label, test_img + img_label)
    shutil.move(npy_dir + npy_label, test_npy + npy_label)













