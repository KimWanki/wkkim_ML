import tensorflow as tf

#vgg


# def cnn_vol2(input_date,filter_count):
#     net = tf.layers.conv2d(inputs=input_date, filters=filter_count, kernel_size=(3, 3), strides=(1, 1), padding='same',
#                              kernel_initializer=kernel_init)
#     net = tf.layers.batch_normalization(net)
#     net = tf.nn.relu(net)
#
#     net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='same')
#     return net

def vgg(input_model,batch_bool):
# TODO : initializer 마다 초기값을 조금 다르게 설정한다.
# he initializer 가중치 초기화 -> relu
    kernel_init = tf.variance_scaling_initializer(scale=2.0)

    # xavier initializer -> sigmoid()
    #kernel_init = tf.contrib.layers.xavier_initializer()

    #tf.constant(3)
    #(10,,,3)
    filter_count = 30
    # input = tf.placeholder(dtype=tf.float32,shape=(10,,,3))
    # batch_bool = tf.placeholder(dtype=tf.bool)

    #정규
    layer = input_model/255.0
    print(layer.shape)
    layer = tf.layers.conv2d(inputs=layer, filters=filter_count, kernel_size=(3, 3), strides=(1,1), padding='same',
                     kernel_initializer=kernel_init)
    # strides => 1, 1
    print(layer.shape)
    #filter : 특성을 잡아낸다. feature map()
    #padding : input output 크기를 같게 하기 위해 padding을 준다.

    # 배치 정규화 >> 성능 차이가 크다.
    # trainging -> 곱셈을 할수록 정보가 날라간다
    layer = tf.layers.batch_normalization(layer, training=batch_bool)
    layer = tf.nn.relu(layer)
    print(layer.shape)
    # TODO : 논문 정보에 따른 학습 순서 지정? 2* conv -> 한 레이어에서 conv를 두행 번 진
    # layer = tf.layers.conv2d(inputs=layer, filters=30, kernel_size=(3, 3), strides=(1,1), padding='same',
    #                  kernel_initializer=kernel_init)
    # # strides => 1, 1
    # print(layer.shape)
    # #filter : 특성을 잡아낸다. feature map()
    # #padding : input output 크기를 같게 하기 위해 padding을 준다.
    #
    # # 배치 정규화 >> 성능 차이가 크다.
    # layer = tf.layers.batch_normalization(layer)
    # layer = tf.nn.relu(layer)
    # print(layer.shape)

    # TODO : conv_layer ->  batch -> relu : 이 성능이 더 좋았다. 데이터 마다 다름.

    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
    print(layer.shape)


    layer = tf.layers.conv2d(inputs=layer, filters=filter_count*2, kernel_size=(3, 3), strides=(1,1), padding='same',
                     kernel_initializer=kernel_init)
    # strides => 1, 1

    print(layer.shape)
    #filter : 특성을 잡아낸다. feature map()
    #padding : input output 크기를 같게 하기 위해 padding을 준다.
    # 배치 정규화 >> 성능 차이가 크다.
    layer = tf.layers.batch_normalization(layer,training=batch_bool)
    layer = tf.nn.relu(layer)
    print(layer.shape)
    # TODO : conv_layer ->  batch -> relu : 이 성능이 더 좋았다. 데이터 마다 다름.
    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
    print(layer.shape)



    layer = tf.layers.conv2d(inputs=layer, filters=filter_count*(2**2), kernel_size=(3, 3), strides=(1,1), padding='same',
                     kernel_initializer=kernel_init)
    # strides => 1, 1

    print(layer.shape)
    #filter : 특성을 잡아낸다. feature map()
    #padding : input output 크기를 같게 하기 위해 padding을 준다.

    # 배치 정규화 >> 성능 차이가 크다.
    layer = tf.layers.batch_normalization(layer,training=batch_bool)
    layer = tf.nn.relu(layer)
    print(layer.shape)
    # TODO : conv_layer ->  batch -> relu : 이 성능이 더 좋았다. 데이터 마다 다름.

    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
    print(layer.shape)



    layer = tf.layers.conv2d(inputs=layer, filters=filter_count*(2**3), kernel_size=(3, 3), strides=(1,1), padding='same',
                     kernel_initializer=kernel_init)
    # strides => 1, 1

    print(layer.shape)
    #filter : 특성을 잡아낸다. feature map()
    #padding : input output 크기를 같게 하기 위해 padding을 준다.

    # 배치 정규화 >> 성능 차이가 크다.
    layer = tf.layers.batch_normalization(layer,training=batch_bool)
    layer = tf.nn.relu(layer)
    print(layer.shape)
    # TODO : conv_layer ->  batch -> relu : 이 성능이 더 좋았다. 데이터 마다 다름.

    # TODO : 이미지를 줄인다. max pooling 잘 안씀, average 또는 strides 값을 변경.
    layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
    print(layer.shape)

    #인풋을 뒤에 작성
    net = tf.keras.layers.Dense(25088,kernel_initializer=kernel_init)(layer)
    net = tf.keras.layers.Dense(4096,kernel_initializer=kernel_init)(layer)
    net = tf.keras.layers.Dense(3,kernel_initializer=kernel_init)(layer)
    #softmax : 확류 합이 1이 되도록 sclae




