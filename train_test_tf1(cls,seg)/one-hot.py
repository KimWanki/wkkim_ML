import numpy as np
import tensorflow as tf
import os
import random
import cv2
import imutils
import math
import matplotlib.pyplot as plt
# kernel_init = tf.variance_scaling_initializer(scale=2.0)
# input_layer = tf.placeholder(dtype=tf.float32,shape=(10,,,3))
# layer = tf.layers.conv2d(inputs=input_layer, filters=30,kerner_size=(3,3), strides=(1,1), padding='same',
#                  kernel_initializer=kernel_init)
# layer = tf.layers.batch_normalization(layer)
# layer = tf.nn.relu(layer)
# strides => 1, 1

#filter : 특성을 잡아낸다. feature map()
#padding : input output 크기를 같게 하기 위해 padding을 준다.

# 배치 정규화 >> 성능 차이가 크다.

def vgg(input_model, batch_bool):
    # TODO : initializer 마다 초기값을 조금 다르게 설정한다.
    # he initializer 가중치 초기화 -> relu
    kernel_init = tf.variance_scaling_initializer(scale=2.0)

    # xavier initializer -> sigmoid()
    # kernel_init = tf.contrib.layers.xavier_initializer()

    # tf.constant(3)
    # (10,,,3)
    filter_count = 30
    # input = tf.placeholder(dtype=tf.float32,shape=(10,,,3))
    # batch_bool = tf.placeholder(dtype=tf.bool)

    # 정규
    layer = input_model / 255.0
    # print(layer.shape)
    layer = tf.layers.conv2d(inputs=layer, filters=filter_count, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=kernel_init)
    # strides => 1, 1
    # print(layer.shape)
    # filter : 특성을 잡아낸다. feature map()
    # padding : input output 크기를 같게 하기 위해 padding을 준다.
    # 배치 정규화 >> 성능 차이가 크다.
    # trainging -> 곱셈을 할수록 정보가 날라간다
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)
    # print(layer.shape)
    # TODO : 논문 정보에 따른 학습 순서 지정? 2* conv -> 한 레이어에서 conv를 두행 번 진
    layer = tf.layers.conv2d(inputs=layer, filters=30, kernel_size=(3, 3), strides=(1,1), padding='same',
                     kernel_initializer=kernel_init)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.nn.relu(layer)
    # TODO : conv_layer ->  batch -> relu : 이 성능이 더 좋았다. 데이터 마다 다름.

    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
    print(layer.shape)

    layer = tf.layers.conv2d(inputs=layer, filters=filter_count * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=kernel_init)
    # strides => 1, 1
    # filter : 특성을 잡아낸다. feature map() # padding : input output 크기를 같게 하기 위해 padding을 준다.
    # 배치 정규화 >> 성능 차이가 크다.
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)

    # TODO : conv_layer -> batch -> relu : 이 성능이 더 좋았다. 데이터 마다 다름.
    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
    print(layer.shape)

    layer = tf.layers.conv2d(inputs=layer, filters=filter_count * (2 ** 2), kernel_size=(3, 3), strides=(1, 1),
                             padding='same',
                             kernel_initializer=kernel_init)
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)
    layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
    layer = tf.layers.conv2d(inputs=layer, filters=filter_count * (2 ** 3), kernel_size=(3, 3), strides=(1, 1),
                             padding='same',
                             kernel_initializer=kernel_init)
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)
    print(layer.shape)

    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
    layer = tf.keras.layers.Flatten()(layer)
    # 인풋을 뒤에 작성
    net = tf.keras.layers.Dense(2048, kernel_initializer=kernel_init)(layer)
    net = tf.keras.layers.Dense(1024, kernel_initializer=kernel_init)(net)
    net = tf.keras.layers.Dense(3, kernel_initializer=kernel_init)(net)
    # softmax : 확류 합이 1이 되도록 sclae
    return net

def img_loader(image_list):
    batch_img_list = []
    for i in image_list:
        input = cv2.imread(i)

        flip_ran = random.randint(0, 3)
        if flip_ran == 0:
            input = cv2.flip(input, 0)
        elif flip_ran == 1:
            input = cv2.flip(input, 1)
        elif flip_ran == 2:
            input = cv2.flip(input, 1)
            input = cv2.flip(input, 0)

        rotate_v = random.randint(-30, 30)
        # TODO : 이미지에서 몇 펴센트 shift()를 진행할지를 결정. # TODO : 얼마나 움직이는지는 이중 랜덤으로 결정해야한다.
        shift_y = input.shape[0] * 0.1
        random_y = random.randint(-shift_y, shift_y)
        shift_x = input.shape[1] * 0.1
        random_x = random.randint(-shift_x, shift_x)

        input = imutils.translate(input, random_x, random_y)
        input = imutils.rotate(input, rotate_v)
        # 모델과 데이터 수 , 전처리 + 이미지를 어떻게 집어넣느냐가 모델 정확성에 정말 많은 부분을 차지한다.

        input = cv2.resize(input, (56, 56))

        # hyper parameter >> 어떤 데이터에서 어떤 모델은 어떨때 가장 좋은지를 계속 피드백.

        batch_img_list.append(input)

    # dtype - int 로 하면 error 발생된다.
    return np.array(batch_img_list, dtype=np.float32)

# TODO : 이미지를 10장씩 가져온다.
def next_batch(data_list, mini_batch_size, next_cnt):
    cnt = mini_batch_size * next_cnt
    batch_list = data_list[cnt:cnt+mini_batch_size]
    return batch_list


# 이미지 전체를 불러와서 일부를 넣는다.
# 폴더 안에 있는 모든 사진 파일을 불러온다.
# TODO : 파일 불러오기. 파일 내 이미지에 정답 할
file_path = "/Users/kimwanki/Downloads/example/"
folder_name = ["A", "B", "C"]
image_list = []
label_list = []
index = 0
for i in folder_name:
    image_path = file_path + i
    _image_list = [file_path + i + '/' + j for j in os.listdir(image_path)]
    _label_list = [index for j in range(len(_image_list))]
    image_list += _image_list
    label_list += _label_list
    index+=1


# TODO : 편향된 학습을 막기 위해 학습의 순서를 뒤섞는다.
# print("shuffle : seed(random)")
# print(image_list)
# print(label_list)

random_variable = random.randint
#학습 순서를 섞는다.
random.seed(random_variable)
random.shuffle(image_list)
random.seed(random_variable)
random.shuffle(label_list)

# print("Result")
# print(image_list)
# print(label_list)

# TODO : opencv2.imread를 활용해 이미지 정보를 확인 및 전처리 준비 gray scale, rgb scale etc.
# test = cv2.imread(image_list[0],0)

#gray scale -> 컬러 채널 0개 gray
# cv2.imread(name_list1[0],0)
#y, x, rgb -> 컬러 채널 3개 r,g,b
# cv2.imread(name_list1[0],1)
#pixel 수 출력 : 세로, 가로, scale 수(ex. gray : 0(default 생략) , rgb = 3)


# TODO : plt 은 이미지를 시각적으로 보여줄 수 있게 해줌.
#image show // imshow(parameter) -> show()
# plt.imshow(test)
# plt.show()



# TODO :
#전체이미지 리스트 길이/ mini_batch size

batch_size = 10
batch_cnt = math.ceil(len(image_list)/batch_size)
x_data = tf.placeholder(dtype=tf.float32, shape=(None,56,56,3))
y_data = tf.placeholder(dtype=tf.float32, shape=(None,3))
train_bool = tf.placeholder(dtype=tf.bool)
model = vgg(x_data,train_bool)
print('model.shape', model.shape)
print('y_data.shape', y_data.shape)
prob = tf.nn.softmax(model)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=model))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    epoch = 50
    for one_epoch in range(epoch):
        total_loss = 0

        for i in range(batch_cnt):
            x_one_batch = next_batch(image_list,batch_size,i)
            # ex)
            x_one_batch = img_loader(x_one_batch)
            y_one_batch = next_batch(label_list,batch_size,i)
            # ex) [0 , 1, 1, ,1, 1,  ] -> one_hot encoding
            y_one_batch = np.eye(3)[len(set(label_list))]
            batch_loss, batch_prob = sess.run([loss,prob], feed_dict={x_data:x_one_batch, y_data:y_one_batch, train_bool:True})
            print(batch_prob,y_one_batch)
            total_loss += batch_loss
        epoch_loss = total_loss/batch_cnt
        print(epoch_loss)



# linux cuda install 확

# name_list1 = [address1 + '/' + i for i in os.listdir(address1)]
# lis1_label = [0 for i in range(len(name_list1))]
# name_list2 = [address2 + '/' + i for i in os.listdir(address2)]
# lise2_label = [1 for i in range(len(name_list2))]
# name_list3 = [address3 + '/' + i for i in os.listdir(address3)]
# lise3_label = [2 for i in range(len(name_list3))]

# y_list1 = lis1_label + lise2_label + lise3_label
# image_list1 = name_list1 + name_list2 + name_list3

# print(image_list1)
# print(y_list1)


# random.seed(0)
# random.shuffle(image_list1)
# random.seed(0)
# random.shuffle(y_list1)
#
# print(image_list1)
# print(y_list1)

# print(name_list1)
# a = os.listdir(address1)
# b = os.listdir(address2)
# c = os.listdir(address3)
#
# image = a + b + c

#print(a)
#print(a+b+c)

#세로, 가로, 컬러 채널
#print(text.shape)



# TODO : 전처리
#rotate, brightness, move
#
# a = cv2.flip(text,0)
# plt.imshow(a)
# plt.show()
#
# a = cv2.flip(text,1)
# plt.imshow(a)
# plt.show()
#
# #a =cv2.rotate(text,1)#
# #plt.imshow(a)
# #plt.show()
#


# # TODO : rotate()
# rotate_v = random.randint(-30,30)
# first_image = imutils. rotate(test, rotate_v)

# # TODO : 이미지에서 몇 펴센트 shift()를 진행할지를 결정. # TODO : 얼마나 움직이는지는 이중 랜덤으로 결정해야한다.
# shift_y = test.shape[0] * 0.1
# random_y = random.randint(-shift_y,shift_y)
#
# shift_x = test.shape[1] * 0.1
# random_x = random.randint(-shift_x,shift_x)
#
# first_image = imutils. translate(test, random_x, random_y)
#
# print(rotate_v)
# print(random_y)
# print(random_x)
# print(shift_y, shift_x)


