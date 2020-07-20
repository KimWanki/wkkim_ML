import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[3,4] = 1 # this would set the label of pixel 3,4 as 1
ann_img[0,0] = 2 # this would set the label of pixel 0,0 as 2

# print(ann_img.shape)
# cv2.imwrite("ann_1.png",ann_img)
#
from keras.layers import Conv2D, Dropout, MaxPooling2D, concatenate, UpSampling2D
from keras.models import Model
from tensorflow.keras import Input

# n_classes = 10
#
# img_input = Input(shape=(224,224 , 3 ))
#
# conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
# conv1 = Dropout(0.2)(conv1)
# conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
# pool1 = MaxPooling2D((2, 2))(conv1)
#
# conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
# conv2 = Dropout(0.2)(conv2)
# conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
# pool2 = MaxPooling2D((2, 2))(conv2)
#
# conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
# conv3 = Dropout(0.2)(conv3)
# conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#
# up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
# conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
# conv4 = Dropout(0.2)(conv4)
# conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
#
# up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
# conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
# conv5 = Dropout(0.2)(conv5)
# conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
#
# out = Conv2D(n_classes, (1, 1) , padding='same')(conv5)
#
# from keras_segmentation.models.model_utils import get_segmentation_model, Input
#
# model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model

from keras_segmentation.models.unet import vgg_unet


model = vgg_unet(n_classes=51,  input_height=224, input_width=224)

model.train(
    train_images =  "dataset/images_prepped_train/",
    train_annotations = "dataset/annotations_prepped_train/",
    checkpoints_path = "checkpoints/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset/images_prepped_test/0016E5_07965.png",
    out_fname="output.png"
)
#
# from keras_segmentation.predict import predict
#
# predict(
# 	checkpoints_path="checkpoints/vgg_unet_1",
# 	inp="dataset/images_prepped_test/0016E5_07965.png",
# 	out_fname="output.png"
# )