import numpy as np
import tensorflow as tf
import os
import random
import cv2
import imutils
import math
import matplotlib.pyplot as plt


def vgg(input_model, batch_bool):
    # TODO : initializer 마다 초기값을 조금 다르게 설정한다.
    # he initializer 가중치 초기화 -> relu
    kernel_init = tf.variance_scaling_initializer(scale=2.0)
    # xavier initializer -> sigmoid()
    # kernel_init = tf.contrib.layers.xavier_initializer()
    # tf.constant(3)
    # (10,56,56,3)

    filter_count = 64
    # input = tf.placeholder(dtype=tf.float32,shape=(10,56,56,3))
    # batch_bool = tf.placeholder(dtype=tf.bool)

    # 정규
    layer = input_model / 255.0
    # print(layer.shape)
    layer = tf.layers.conv2d(inputs=layer, filters=filter_count, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=kernel_init)

    # print(layer.shape)
    # filter : 특성을 잡아낸다. feature map()
    # padding : input output 크기를 같게 하기 위해 padding을 준다.
    # 배치 정규화 >> 성능 차이가 크다.
    # training -> 곱셈을 할수록 정보가 날라간다

    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)
    # print(layer.shape)
    # TODO : 논문 정보에 따른 학습 순서 지정? 2* conv -> 한 레이어에서 conv를 두행 번 진
    layer = tf.layers.conv2d(inputs=layer, filters=filter_count, kernel_size=(3, 3), strides=(1, 1), padding='same',
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

def test_img_loader(image_list):
    batch_img_list = []
    for i in image_list:
        input = cv2.imread(i)
        input = cv2.resize(input, (56, 56))

        # hyper parameter >> 어떤 데이터에서 어떤 모델은 어떨때 가장 좋은지를 계속 피드백.

        batch_img_list.append(input)

    # dtype - int 로 하면 error 발생된다.
    return np.array(batch_img_list, dtype=np.float32)

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
        shift_y = input.shape[0] * 0.1
        shift_y = math.ceil(shift_y)
        random_y = random.randint(-shift_y, shift_y)
        shift_x = input.shape[1] * 0.1
        shift_x = math.ceil(shift_x)
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
    batch_list = data_list[cnt:cnt + mini_batch_size]
    return batch_list


# 이미지 전체를 불러와서 일부를 넣는다.
# 폴더 안에 있는 모든 사진 파일을 불러온다.
# TODO : 파일 불러오기. 파일 내 이미지에 정답 할
# file_path = "/Users/kimwanki/Downloads/example/"
# folder_name = ["A", "B", "C"]
file_path = "/Users/kimwanki/developer/testcase/"
folder_name = ["sweetpo", "straw", "potato"]

test_path = "/Users/kimwanki/developer/testcase/test"
image_list = []
label_list = []


index = 0

for i in folder_name:
    image_path = file_path + i
    _image_list = [file_path + i + '/' + j for j in os.listdir(image_path)]
    _label_list = [index for j in range(len(_image_list))]
    image_list += _image_list
    label_list += _label_list
    index += 1

test_list = ['/Users/kimwanki/developer/testcase/test/딸기3.jpg', '/Users/kimwanki/developer/testcase/test/딸기2.jpg', '/Users/kimwanki/developer/testcase/test/딸기5.jpg', '/Users/kimwanki/developer/testcase/test/딸기4.jpg', '/Users/kimwanki/developer/testcase/test/딸기6.jpg', '/Users/kimwanki/developer/testcase/test/고구마79.jpg', '/Users/kimwanki/developer/testcase/test/고구마50.jpg', '/Users/kimwanki/developer/testcase/test/고구마46.jpg', '/Users/kimwanki/developer/testcase/test/고구마54.jpg', '/Users/kimwanki/developer/testcase/test/고구마37.jpg']
print(len(test_list))
print(test_list)
test_label = [1,1,1,1,1,0,0,0,0,0]
print(len(test_label))
print(test_label)


# TODO : 편향된 학습을 막기 위해 학습의 순서를 뒤섞는다.
# print("shuffle : seed(random)")
# print(image_list)
print(label_list)

random_variable = random.randint
# 학습 순서를 섞는다.
random.seed(random_variable)
random.shuffle(image_list)
random.seed(random_variable)
random.shuffle(label_list)

# TODO : opencv2.imread를 활용해 이미지 정보를 확인 및 전처리 준비 gray scale, rgb scale etc.
# test = cv2.imread(image_list[0],0)


# TODO : plt 은 이미지를 시각적으로 보여줄 수 있게 해줌.
# image show // imshow(parameter) -> show()
# plt.imshow(test)
# plt.show()

# TODO : #전체이미지 리스트 길이/ mini_batch size

batch_size = 10
batch_cnt = math.ceil(len(image_list) / batch_size)

x_data = tf.placeholder(dtype=tf.float32, shape=(None, 56, 56, 3))
y_data = tf.placeholder(dtype=tf.float32, shape=(None, 3))

train_bool = tf.placeholder(dtype=tf.bool)

model = vgg(x_data, train_bool)
print('model.shape', model.shape)
print('y_data.shape', y_data.shape)

global_step = tf.Variable(0, trainable=False, name='global_step')
prob = tf.nn.softmax(model)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=model))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# with tf.control_dependencies(update_ops):
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()

display_step = 1
with tf.Session() as sess:
    ############# tf.global_variables() 은 사용 안해도 무방

    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    epoch = 20
    for one_epoch in range(epoch):
        total_loss = 0
        ############# avg_cost = 0.
        ############# avg_accuracy = 0
        for i in range(batch_cnt):
            avg_cost = 0. ############ batch 단위에 대한 loss를 알고 싶으면 여기에 두는 것이 맞는데
                          ############ batch의 평균 loss를 알고 싶으면 228 line 주석 부분에 avg_cost를 넣어야함
            x_one_batch = next_batch(image_list, batch_size, i)
            x_one_batch = img_loader(x_one_batch)
            y_one_batch = next_batch(label_list, batch_size, i)
            # ex) [0 , 1, 1, ,1, 1,  ] -> one_hot encoding
            y_one_batch = np.eye(len(set(label_list)))[y_one_batch]
            # [len(set(label_list))]
            _, batch_loss, batch_prob = sess.run([optimizer,loss, prob],
                                              feed_dict={x_data: x_one_batch, y_data: y_one_batch, train_bool: True})
            # batch_prob,batch_loss  = sess.run([optimizer, loss],feed_dict={x_data: x_one_batch, y_data: y_one_batch, train_bool: True})
            print(batch_prob, y_one_batch)
            avg_cost += batch_loss / batch_cnt
            # total_loss += batch_loss

            ############ correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_data, 1))  ###### model에 batch_prob
            ############ avg_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        epoch_loss = total_loss / batch_cnt
        if (one_epoch +1) % display_step == 0:
            print("Epoch",'%04d'%(one_epoch+1), "cost = ","{:.9f}".format(avg_cost))
        # print(epoch_loss)
    print("Optimization Finished!")

    # saver.save(sess, './model/cnn.ckpt', global_step=global_step)

    ################# tf.argmax(model,1)에서 model은 확률값으로 정규화 되지 않은 값을 의미합니다
    ################# 저희가 원하는건 확률값으로 정규화된 값과 정답과 비교를 해야되기 때문에 batch_prob 를 가져와야 합니다
    ################# 또한 batch 단위로 정확도를 보고 싶으면 loss와 마찬가지로 batch for문에 아래 코드가 들어가야합니다
    ################# tf는 길을 만드는 것이기 때문에 for 문에 들어가있으면 tensor를 계속 만들어 나중에 모델이 터집니다
    ################# for문 밖에 아래 코드를 만들던가 numpy를 통해 계산하면 됩니다
    ################# 저같은 경우 numpy를 통해 계산을 진행합니다
    ################# 코드 수정은 제가 보내드린 코드를 참고하셔서 수정하시면 될 것 같습니다
    correct_prediction = tf.equal(tf.argmax(batch_prob,1),tf.argmax(y_data,1))   ###### model에 batch_prob
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    test_list = test_img_loader(test_list)
    test_label = np.eye(3)[test_label]
    print("accuracy", sess.run(accuracy,feed_dict={x_data:test_list, y_data:test_label, train_bool:False}))
    print(accuracy)

# prediction = tf.argmax(model, 0)
# target = tf.argmax(y_data, 0)
# print('예측값:', sess.run(prediction, feed_dict={x_data: x_one_batch}))
# print('실제값:', sess.run(target, feed_dict={y_data: y_one_batch}))
#
# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.case(is_correct, tf.float32))
# print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={x_data: x_one_batch, y_data: y_one_batch}))



init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    for k in range(epoch):
        print('epoch : ', k)
        val_accuracy = 0
        train_accuracy = 0

        for i in range(batch_size):
            one_batch_X_list = next_batch(total_train_X, mini_batch_size, batch_cnt)
            print('one_batch_X_list', one_batch_X_list)
            one_batch_Y = next_batch(total_train_Y, mini_batch_size, batch_cnt)
            one_batch_Y = one_hot_Y(one_batch_Y, 2)
            one_batch_X = img_loader(one_batch_X_list, 45, 0.1, True)

            feed_dict = {X_: one_batch_X, Y_: one_batch_Y, training: True}
            _, loss, prob, X, Y = sess.run([optimizer, cost, probability, X_, Y_], feed_dict=feed_dict)

            ###########################
            result = np.argmax(prob, axis=1)
            label = np.argmax(Y, axis=1)
            print('True_Y', label)
            print('predict_Y', result)

            one_batch_train_acc = np.mean(np.equal(result, label).astype(int))
            train_accuracy += one_batch_train_acc
            ###########################

        for i in range(val_batch_size):
            one_batch_X_list = next_batch(total_val_X, mini_batch_size, batch_cnt)
            one_batch_Y = next_batch(total_val_Y, mini_batch_size, batch_cnt)
            one_batch_Y = one_hot_Y(one_batch_Y, 2)
            one_batch_X = img_loader(one_batch_X_list, 0, 0, aug=False)

            feed_dict = {X_: one_batch_X, Y_: one_batch_Y, training: False}
            loss, prob, label = sess.run([cost, probability, Y_], feed_dict=feed_dict)

            ###########################
            result = np.argmax(prob, axis=1)
            label = np.argmax(label, axis=1)

            val_accuracy += np.mean(np.equal(result, label).astype(int))

            ckpt_path = saver.save(sess, "/gdrive/My Drive/data/dog&cat/ckpt/dog_cat.ckpt")
            ###########################