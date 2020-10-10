import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import warnings
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from keras.utils import Sequence
from imgaug import augmenters as iaa 
BATCH_SIZE = 96
seq = iaa.Sequential([
                                iaa.Affine(scale={"x": (0.8,1.2),"y": (0.8,1.2)},
                                translate_percent = {"x": (-0.2,0.2),"y": (-0.2,0.2)},
                                rotate = (-25,25))
                     ])
def big_XCEPTION(input_shape):
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    x = Dropout(0.20)(x)
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = Dropout(0.20)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    residual = Conv2D(512, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = Dropout(0.20)(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])
    for i in range(2):
        residual = x
        prefix = 'block' + str(i + 5)
        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = Dropout(0.20)(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
        x = layers.add([x, residual])
    x = Conv2D(1, (3, 3),padding='same')(x)
    x = Flatten()(x)
    predictions_g = Dense(2, activation="softmax")(x)
    predictions_a = Dense(7, activation="softmax")(x)
    model = Model(img_input, outputs=[predictions_g, predictions_a])
    return model
model = big_XCEPTION((128,128,3))
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
print (model.summary())

class DataGenerator(Sequence):
    def __init__(self,csv_file):
        data = pd.read_csv(csv_file).values
        self.labels = data[:,1:]
        self.img_path =data[:,0]
    def __len__(self):
        return math.ceil(self.labels.shape[0]) // BATCH_SIZE
    def __getitem__(self, idx):
        batch_paths = self.img_path[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_labels = self.labels[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE,:]
        lables_gender = np.zeros((batch_paths.shape[0], 2), dtype=np.int8)
        lables_age = np.zeros((batch_paths.shape[0], 7), dtype=np.int8)
        batch_images = np.zeros((batch_paths.shape[0], 128, 128, 3), dtype=np.float32)
        try:
            for i, f in enumerate(batch_paths):
                img = cv2.imread(f)
                img = cv2.resize(img, (128, 128),interpolation=cv2.INTER_AREA)
                img = img/255.0
                img = img -0.5
                img = img*2.0
                batch_images[i] = img
            for i, f in enumerate(batch_labels):
                lables_gender[i][f[0]] = 1
                lables_age[i][f[1]] = 1
        except:
            pass
        return seq(images = batch_images), [lables_gender, lables_age]
train_datagen = DataGenerator("/home/subodh/Put_Mask/age_n_gen/training_annotations.csv")
val_datagen = DataGenerator("/home/subodh/Put_Mask/age_n_gen/validation_annotations.csv")

def lr_sch(epoch):
    return 0.0001

lr_scheduler=LearningRateScheduler(lr_sch)
lr_reducer=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,mode='max',min_lr=1e-3)
checkpoint=ModelCheckpoint('model-d-{val_loss:.2f}.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
model_details=model.fit_generator(generator=train_datagen,epochs=30,shuffle=True,validation_data=val_datagen,callbacks=[lr_scheduler,lr_reducer,checkpoint],verbose=1)
model.save('/home/subodh/Put_Mask/model_d_final.h5')