import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import math
import random
import os
import tensorflow as tf
from segmentation import vgg




img = cv2.imread('/Users/kimwanki/PycharmProjects/segmentation/labelimg/SegmentationClass_result/000032.png')
plt.imshow(img)
# plt.show()
print(img.shape)
width, height = img.shape[1],img.shape[0]
print(width, height)


# background = np.zeros_like(img)
background_person = np.zeros(shape = (img.shape[0],img.shape[1]))
background_plain = np.zeros(shape = (img.shape[0],img.shape[1]))


background_person[(img[:,:,0] == 128) & (img[:,:,1] == 128) & (img[:,:,2] == 192)] = 1
background_plain[(img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 128)] = 1
plt.imshow(background_plain)
plt.show()

img = np.stack([background_person,background_plain],axis=2)

def img_augmentation(input, flip_ran, rotate_v):
    if flip_ran == 0:
        input = cv2.flip(input, 0)
    elif flip_ran == 1:
        input = cv2.flip(input, 1)
    elif flip_ran == 2:
        input = cv2.flip(input, 1)
        input = cv2.flip(input, 0)

    shift_y = input.shape[0] * 0.1
    shift_y = math.ceil(shift_y)
    random_y = random.randint(-shift_y, shift_y)
    shift_x = input.shape[1] * 0.1
    shift_x = math.ceil(shift_x)
    random_x = random.randint(-shift_x, shift_x)

    input = imutils.translate(input, random_x, random_y)
    input = imutils.rotate(input, rotate_v)
    input = cv2.resize(input, (224, 224))

    return input

    # TODO : 이미지를 10장씩 가져온다.
def next_batch(data_list, mini_batch_size, next_cnt):
    cnt = mini_batch_size * next_cnt
    batch_list = data_list[cnt:cnt + mini_batch_size]
    return batch_list

    # 모델과 데이터 수 , 전처리 + 이미지를 어떻게 집어넣느냐가 모델 정확성에 정말 많은 부분을 차지한다.


def img_loader(img_name_list, mask_name_list, train):
    batch_img_list = []
    batch_mask_list = []
    for train_img, mask_img in zip(img_name_list, mask_name_list):
        train_img = cv2.imread(train_img)
        mask_img = cv2.imread(mask_img)

        if (train == 1):
            flip_ran = random.randint(0, 3)
            rotate_v = random.randint(-30, 30)
            train_img = img_augmentation(train_img, flip_ran, rotate_v)
            mask_img = img_augmentation(mask_img, flip_ran, rotate_v)

            _, mask_img = cv2.threshold(mask_img, 0, 1, cv2.THRESH_BINARY)
        else:
            train_img = cv2.resize(train_img, (224, 224))
            mask_img = cv2.resize(mask_img, (224, 224))
            _, mask_img = cv2.threshold(mask_img, 0, 1, cv2.THRESH_BINARY)
        # hyper parameter >> 어떤 데이터에서 어떤 모델은 어떨때 가장 좋은지를 계속 피드백.

        batch_img_list.append(train_img)
        batch_mask_list.append(mask_img)

    # dtype - int 로 하면 error 발생된다.
    return np.array(batch_img_list, dtype=np.float32), np.array(batch_mask_list, dtype=np.float32)


# 이미지 전체를 불러와서 일부를 넣는다.
# 폴더 안에 있는 모든 사진 파일을 불러온다.
# TODO : 파일 불러오기. 파일 내 이미지에 정답 할
# file_path = "/Users/kimwanki/Downloads/example/"
# folder_name = ["A", "B", "C"]

img_path = '/Users/kimwanki/PycharmProjects/segmentation/labelimg/trainimg'
mask_path = '/Users/kimwanki/PycharmProjects/segmentation/labelimg/SegmentationClass_result'

img_list = os.listdir(img_path)
mask_list = os.listdir(mask_path)

mask_list.remove('.DS_Store')
print(len(mask_list))

img_idx = int(len(img_list) * 0.8)

img_train = img_list[:img_idx]
img_val = img_list[img_idx:]

mask_train = mask_list[:img_idx]
mask_val = mask_list[img_idx:]

import math
BATCH_SIZE = 5
BATCH_CNT = math.ceil(len(img_train)/BATCH_SIZE)


img_data = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
mask_data = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 2))

train_bool = tf.placeholder(dtype=tf.bool)
model = vgg(img_data, train_bool)

#with로 대체
#sess = tf.Session()
#sess.close
model = vgg(input)
with tf.Session() as Sess:
    for i in range(BATCH_CNT):
        _img_list = next_batch(img_train, BATCH_SIZE, i)
        _mask_list = next_batch(mask_train, BATCH_SIZE, i)

        batch_img_list, batch_mask_list = img_loader(_img_list,_mask_list,1)



