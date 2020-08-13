#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:27:53 2020

@author: wkkim
"""

import keras
import os
import innvestigate
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

DIR_MODEL_SEG = 'C:/Users/USER/Desktop/wkkim/2020intern_1/medicalAI/segmentation_models_keras/model'
DIR_MODEL_CLS = 'C:/Users/USER/Desktop/wkkim/2020intern_1/medicalAI/classification_models_keras/model'

DIR_DATA = 'C:/Users/USER/Desktop/wkkim/test'

# inceptionv3 , densenet121, resnet50, vgg16
MODEL_TYPE = 'segmentation'
BACKBONE = 'vgg16'
CASES = ['Adeno', 'HGD', 'LGD', 'Normal']

if not os.path.isfile(DIR_DATA + '/Adeno_Cls/'):
    os.mkdir(DIR_DATA + '/Adeno_Cls/')
    os.mkdir(DIR_DATA + '/HGD_Cls/')
    os.mkdir(DIR_DATA + '/LGD_Cls/')
    os.mkdir(DIR_DATA + '/Normal_Cls/')

def get_data(path_img, path_msk):
    # read data
    image = cv2.imread(path_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(self.masks_fps[i], 0)
    if path_msk:
        mask = np.load(path_msk, 0).astype(np.uint8)
    else:
        mask = np.zeros(image.shape[:2]).astype(np.uint8)
    h, w = mask.shape[0], mask.shape[1]
    # input size
    if h > w:
        # image_bg = np.zeros(shape=(h, math.ceil((h - w) / 2), 3), dtype=np.uint8)
        mask_bg = np.zeros(shape=(h, math.ceil((h - w) / 2)), dtype=np.uint8)
        # image = cv2.hconcat([image_bg, image, image_bg])
        mask = cv2.hconcat([mask_bg, mask, mask_bg])
    elif h < w:
        # image_bg = np.zeros(shape=(math.ceil((w - h) / 2), w, 3), dtype=np.uint8)
        mask_bg = np.zeros(shape=(math.ceil((w - h) / 2), w), dtype=np.uint8)
        # image = cv2.vconcat([image_bg, image, image_bg])
        mask = cv2.vconcat([mask_bg, mask, mask_bg])

    image = cv2.resize(image, (320, 320))
    mask = cv2.resize(mask, (320, 320))

    return image, mask


if MODEL_TYPE == 'classification':
    from classification_models.keras import Classifiers
    from keras.layers import GlobalAveragePooling2D, Dense
    from keras.models import Model

    base, preprocess_input = Classifiers.get(BACKBONE)

    n_classes = 4

    # build model
    base_model = base(input_shape=(320, 320, 3), weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = True

    global_avrg_pool = GlobalAveragePooling2D()(base_model.output)
    fc_1 = Dense(1024, activation='relu', name='fc_1')(global_avrg_pool)
    predictions = Dense(n_classes, activation='softmax', name='predictions')(fc_1)
    model = Model(inputs=base_model.input, outputs=predictions)

    if BACKBONE == 'densenet121':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'DenseNet121_rescale.h5'))
        # model.load_weights(os.path.join(DIR_MODEL_CLS, 'DenseNet121_320.h5'))
    elif BACKBONE == 'resnet50':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'ResNet50V1_rescale.h5'))
    elif BACKBONE == 'vgg16':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'vgg_rescale.h5'))
    elif BACKBONE == 'inceptionv3':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'InceptionV3_rescale.h5'))

elif MODEL_TYPE == 'segmentation':
    import segmentation_models as sm

    BATCH_SIZE = 5
    CLASSES = ['polyp']
    LR = 0.0001
    EPOCHS = 100

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1
    activation = 'sigmoid'

    # create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(320, 320, 3))
    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    if BACKBONE == 'densenet121':
        model.load_weights(os.path.join(DIR_MODEL_SEG, 'densenet121_up_1.h5'))
    elif BACKBONE == 'resnet50':
        model.load_weights(os.path.join(DIR_MODEL_SEG, 'resnet50_up_1.h5'))
    elif BACKBONE == 'vgg16':
        model.load_weights(os.path.join(DIR_MODEL_SEG, 'vgg16_up_1.h5'))
    elif BACKBONE == 'inceptionv3':
        model.load_weights(os.path.join(DIR_MODEL_SEG, 'InceptionV3_320.h5'))

# print(model.summary())
if MODEL_TYPE == 'segmentation':
    # Segmentation Model
    # model_wo = innvestigate.utils.model_wo_softmax(model)
    # selected_seg = keras.layers.Lambda(lambda x: x[1])(model.layers[-1].output)
    # model.summary()
    # layer = model.get_layer('center_block2_conv').output
    # print(layer)
    # print(model.layers[-3].output)

    selected_seg = keras.layers.Lambda(lambda x: x[:, :, :, :])(model.layers[-1].output)

    # selected_seg = keras.layers.Lambda(lambda x: x[:, :, :, 1])(layer)
    selected_seg = keras.layers.Flatten()(selected_seg)
    model_seg_final = keras.Model(inputs=model.layers[0].input, outputs=[selected_seg])
    analyzer = innvestigate.create_analyzer('gradient', model_seg_final, neuron_selection_mode="all",
                                            allow_lambda_layers=True)
    # analyzer = innvestigate.create_analyzer()

elif MODEL_TYPE == 'classification':

    # Classification model

    # Stripping the softmax activation from the model
    model_wo = innvestigate.utils.model_wo_softmax(model)
    # analyzer = innvestigate.create_analyzer('gradient',model_wo, neuron_selection_mode="index")
    analyzer = innvestigate.create_analyzer('gradient', model_wo, neuron_selection_mode="max_activation")

    # no difference
    # model_wo.summary()
    # selected_cls = keras.layers.Lambda(lambda x: x[:, :, :, 1])(model_wo.layers[-1].output)
    # selected_cls = keras.layers.Flatten()(selected_cls)
    # selected_cls = keras.layers.Lambda(lambda x: x[3])(model_wo.layers[-1].output)
    # selected_cls = keras.layers.Lambda(lambda x: keras.backend.max(x, axis=-1))(model_wo.layers[-1].output)
    #
    # model_cls_final = keras.Model(inputs=model_wo.layers[:,:,:,0].input, outputs=[selected_cls])
    # analyzer = innvestigate.create_analyzer('gradient', model_cls_final, neuron_selection_mode="max_activation", allow_lambda_layers=True)
else:
    raise ValueError('Model type def')

for CASE in CASES:
    test_path = os.listdir(os.path.join(DIR_DATA, CASE))
    path_img_cancer = glob.glob(os.path.join(DIR_DATA, CASE + '/*'))
    path_msk_cancer = glob.glob(os.path.join(DIR_DATA, CASE + '_mask/*'))
    for i in range(len(test_path)):
        img_cancer, msk_cancer = get_data(path_img_cancer[i], path_msk_cancer[i])
        # model.predict(img_cancer[np.newaxis, :, :])

        analysis = analyzer.analyze(img_cancer[np.newaxis, :, :])
        # analysis = analyzer.analyze(img_cancer[np.newaxis, :, :])
        analysis = np.clip(analysis, np.percentile(analysis, 6), np.percentile(analysis, 99.9))

        analysis -= np.min(analysis)
        analysis /= np.max(analysis)

        heatmap = analysis[0, :, :, :] * 255
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        cv2.imwrite(DIR_DATA+'/'+CASE+'_Cls/'+ test_path[i].replace('tif', 'bmp').replace('bmp', 'jpg').replace('jpg', 'png'), heatmap)
