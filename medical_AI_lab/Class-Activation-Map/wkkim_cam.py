#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:27:53 2020

"""

# import segmentation_models as sm
import keras
import os
import innvestigate
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

DIR_MODEL_SEG = 'C:/Users/USER/PycharmProjects/2020intern_1/medicalAI/segmentation_models_keras/model'
DIR_MODEL_CLS = 'C:/Users/USER/PycharmProjects/2020intern_1/medicalAI/classification_models_keras/model'

DIR_DATA = 'C:/Users/USER/Desktop/test'


MODEL_TYPE = 'classification'
BACKBONE = 'vgg16'


def get_data(path_img, path_msk):
    # read data
    image = cv2.imread(path_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(self.masks_fps[i], 0)
    if path_msk:
        mask = np.load(path_msk, 0).astype(np.uint8)
    else:
        mask = np.zeros(image.shape[:2]).astype(np.uint8)
    h, w = image.shape[0], image.shape[1]

    if h > w:
        image_bg = np.zeros(shape=(h, math.ceil((h - w) / 2), 3), dtype=np.uint8)
        mask_bg = np.zeros(shape=(h, math.ceil((h - w) / 2)), dtype=np.uint8)
        image = cv2.hconcat([image_bg, image, image_bg])
        mask = cv2.hconcat([mask_bg, mask, mask_bg])
    elif h < w:
        image_bg = np.zeros(shape=(math.ceil((w - h) / 2), w, 3), dtype=np.uint8)
        mask_bg = np.zeros(shape=(math.ceil((w - h) / 2), w), dtype=np.uint8)
        image = cv2.vconcat([image_bg, image, image_bg])
        mask = cv2.vconcat([mask_bg, mask, mask_bg])

    image = cv2.resize(image, (320, 320))
    mask = cv2.resize(mask, (320, 320))

    return image, mask


if MODEL_TYPE == 'classification':
    from classification_models import Classifiers
    from keras.layers import GlobalAveragePooling2D, Dense
    from keras.models import Model

    Vgg16, preprocess_input = Classifiers.get('vgg16')

    n_classes = 4

    # build model
    base_model = Vgg16(input_shape=(320, 320, 3), weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = True

    global_avrg_pool = GlobalAveragePooling2D()(base_model.output)
    fc_1 = Dense(1024, activation='relu', name='fc_1')(global_avrg_pool)
    predictions = Dense(n_classes, activation='softmax', name='predictions')(fc_1)
    model = Model(inputs=base_model.input, outputs=predictions)

    if BACKBONE == 'densnet':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'densenet121_320.h5'))
    elif BACKBONE == 'resnet':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'ResNet50V2_320.h5'))
    elif BACKBONE == 'vgg16':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'Vgg16_320.h5'))
    elif BACKBONE == 'inception':
        model.load_weights(os.path.join(DIR_MODEL_CLS, 'InceptionV3_320.h5'))


elif MODEL_TYPE == 'segmentation':
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

    if BACKBONE == 'densnet':
        model.load_weight(os.path.join(DIR_MODEL_SEG, 'densenet121.h5'))
    elif BACKBONE == 'resnet':
        model.load_weight(os.path.join(DIR_MODEL_SEG, 'resnet50.h5'))
    elif BACKBONE == 'vgg':
        model.load_weight(os.path.join(DIR_MODEL_SEG, 'vgg16.h5'))
    elif BACKBONE == 'inception':
        model.load_weight(os.path.join(DIR_MODEL_SEG, 'inceptionv3.h5'))



path_img_normal = glob.glob(os.path.join(DIR_DATA, 'Normal/*'))[0]
path_img_lgd = glob.glob(os.path.join(DIR_DATA, 'LGD/*'))[0]
path_img_hgd = glob.glob(os.path.join(DIR_DATA, 'HGD/*'))[0]
path_img_cancer = glob.glob(os.path.join(DIR_DATA, 'Adeno/*'))[0]

path_msk_normal = False
path_msk_lgd = path_img_lgd.replace('.jpg','.npy')
path_msk_hgd = path_img_hgd.replace('.jpg','.npy')
path_msk_cancer = path_img_cancer.replace('.jpg','.npy')


img_normal, msk_normal = get_data(path_img_normal, path_msk_normal)
img_lgd, msk_lgd = get_data(path_img_lgd, path_msk_lgd)
img_hgd, msk_hgd = get_data(path_img_hgd, path_msk_hgd)
img_cancer, msk_cancer = get_data(path_img_cancer, path_msk_cancer)

if MODEL_TYPE == 'segmentation':
    # Segmentation Model
    analyzer = innvestigate.create_analyzer('lrp.z', model, neuron_selection_mode="all")
elif MODEL_TYPE == 'classification':
    # Classification model
    model_wo = innvestigate.utils.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer('lrp.z', model_wo, neuron_selection_mode="max_activation")
else:
    raise ValueError('Model type def')


analysis = analyzer.analyze(img_normal[np.newaxis, :, :])
analysis = np.clip(analysis, np.percentile(analysis,80), np.percentile(analysis,99.9))
analysis -= np.min(analysis)
analysis /= np.max(analysis)
#analysis = (analysis*255).astype(np.uint8)
fig = plt.figure(1, figsize=(15,4))
plt.subplot(131)
plt.imshow(img_normal)
plt.subplot(132)
plt.imshow(msk_normal)
plt.subplot(133)
plt.imshow(analysis[0, :, :, :])
plt.title('Normal')
plt.show()
plt.close('all')

analysis = analyzer.analyze(img_lgd[np.newaxis, :, :])
analysis = np.clip(analysis, np.percentile(analysis,80), np.percentile(analysis,99.9))
analysis -= np.min(analysis)
analysis /= np.max(analysis)

fig = plt.figure(1, figsize=(15,4))
plt.subplot(131)
plt.imshow(img_normal)
plt.subplot(132)
plt.imshow(msk_normal)
plt.subplot(133)
plt.imshow(analysis[0, :, :, :])
plt.title('LGD')
plt.show()
plt.close('all')


analysis = analyzer.analyze(img_hgd[np.newaxis, :, :])
analysis = np.clip(analysis, np.percentile(analysis,80), np.percentile(analysis,99.9))
analysis -= np.min(analysis)
analysis /= np.max(analysis)
#analysis = (analysis*255).astype(np.uint8)
fig = plt.figure(1, figsize=(15,4))
plt.subplot(131)
plt.imshow(img_hgd)
plt.subplot(132)
plt.imshow(msk_hgd)
plt.subplot(133)
plt.imshow(analysis[0, :, :, :])
plt.title('hgd')
plt.show()
plt.close('all')

analysis = analyzer.analyze(img_cancer[np.newaxis, :, :])
analysis = np.clip(analysis, np.percentile(analysis,80), np.percentile(analysis,99.9))
analysis -= np.min(analysis)
analysis /= np.max(analysis)

#analysis = (analysis*255).astype(np.uint8)
fig = plt.figure(1, figsize=(15,4))
plt.subplot(131)
plt.imshow(img_cancer)
plt.subplot(132)
plt.imshow(msk_cancer)
plt.subplot(133)
plt.imshow(analysis[0, :, :, :])
plt.suptitle('cancer')
plt.show()
plt.close('all')

