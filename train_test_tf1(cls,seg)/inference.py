import numpy as np
import tensorflow as tf
import cv2

def vgg(input_model, batch_bool):
    # TODO : initializer 마다 초기값을 조금 다르게 설정한다.
    kernel_init = tf.variance_scaling_initializer(scale=2.0)
    filter_count = 30
    layer = input_model / 255.0
    layer = tf.layers.conv2d(inputs=layer, filters=filter_count, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=kernel_init)
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)
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
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)

    layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
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

    layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
    layer = tf.keras.layers.Flatten()(layer)
    # 인풋을 뒤에 작성
    net = tf.keras.layers.Dense(2048, kernel_initializer=kernel_init)(layer)
    net = tf.keras.layers.Dense(1024, kernel_initializer=kernel_init)(net)
    net = tf.keras.layers.Dense(3, kernel_initializer=kernel_init)(net)
    # softmax : 확류 합이 1이 되도록 sclae
    return net #마지막 컨볼루션

path = '/Users/kimwanki/developer/testcase/potato/감자6.jpg'
classname = ["sweetpo", "straw", "potato"]

x_data = tf.placeholder(dtype=tf.float32, shape=(None, 56, 56, 3)) #성능 비교시에는 Y값이 필요.
train_bool = tf.placeholder(dtype=tf.bool) # y_data = tf.placeholder(dtype=tf.float32, shape=(None, 3))
model = vgg(x_data, train_bool)
prob = tf.nn.softmax(model)
class_output = tf.argmax(prob,axis = 1)

img = cv2.imread(path)
print(img.shape)
img = cv2.resize(img,(56,56))
img = np.expand_dims(img,axis=0)
print(img.shape)

#tf.placeholder(shape=[None,224,224,3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess,save_path='./model/cnn.ckpt')
    test_prob =sess.run(class_output, feed_dict={x_data:img, train_bool:False})
    print(test_prob)
    print(classname[test_prob[0]])