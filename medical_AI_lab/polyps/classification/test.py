# conda activate wkkim
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from classification_models.keras import Classifiers
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

Vgg16, preprocess_input = Classifiers.get('vgg16')

n_classes = 4

#build model
base_model = Vgg16(input_shape=(320, 320, 3), weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = True

global_avrg_pool = GlobalAveragePooling2D()(base_model.output)
fc_1 = Dense(1024, activation='relu', name='fc_1')(global_avrg_pool)
predictions = Dense(n_classes, activation='softmax', name='predictions')(fc_1)
model = Model(inputs=base_model.input, outputs=predictions)

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    directory ='C:/Users/user/Desktop/wkkim/classification/dataset/comp_test/cls',
    target_size=(320, 320),
    batch_size=10,
    class_mode='categorical',
    shuffle=False
)

from keras import optimizers

model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('C:/Users/user/Desktop/wkkim/2020intern_1/medicalAI/classification_models_keras/model/vgg16_320.h5')

import math

print("-- Predict --")
loss, acc = model.evaluate_generator(test_generator, verbose=1, steps=math.ceil(303/10))
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(loss, acc)
