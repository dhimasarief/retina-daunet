# part of this script was taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation
import argparse
from glob import glob
 
import numpy as np
from PIL import Image
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from numpy import random
import tensorflow as tf
from aug_utils import random_augmentation
from random import randint
import cv2
from keras.regularizers import l2


batch_size = 16
input_shape = (48, 48)
 
def custom_activation(x):
    return K.relu(x, alpha=0.0, max_value=1)
 
 
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def entropy_OTSU_loss(y_true, y_pred):
    bce = losses.binary_crossentropy
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_true))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_true))
    mean_1 = K.sum(pt_1)/(K.sum(y_true)+1E-6)
    mean_0 = K.sum(pt_0)/(K.sum(1-y_true)+1E-6)
    weights_1 = K.sum(y_true)/(K.sum(tf.ones_like(y_true)))
    weights_0 = 1 - weights_1
    var_1 = K.sum((y_true)*(pt_1-mean_1)**2)/(K.sum(y_true)+1E-6)
    var_0 = K.sum((1-y_true)*(pt_0-mean_0)**2)/(K.sum(1-y_true)+1E-6)
    return bce(y_true,y_pred) + weights_0*var_0 + weights_1*var_1 + weights_0*weights_1*(1-(mean_1-mean_0)**2)

smooth = 1.
 
def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    
    x = Conv2D(nb_filter, (1, 1),
               #kernel_initializer="he_uniform",
               padding="same",
               #use_bias=False,
               #kernel_regularizer=l2(weight_decay)
               )(x)
    
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    
    x = Conv2D(nb_filter, (3, 3),
               #kernel_initializer="he_uniform",
               padding="same",
               #use_bias=False,
               #kernel_regularizer=l2(weight_decay)
               )(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x

def transition(x, concat_axis, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1),
               #kernel_initializer="he_uniform",
               padding="same",
               #use_bias=False,
               #kernel_regularizer=l2(weight_decay)
               )(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x

def transition_transpose(x, concat_axis, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1),
               #kernel_initializer="he_uniform",
               padding="same",
               #use_bias=False,
               #kernel_regularizer=l2(weight_decay)
               )(x)
    x = Conv2DTranspose(nb_filter, (2, 2), strides=(2, 2), padding='same')(x)

    return x

def denseblock(x, concat_axis, nb_layers, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x

def get_daunet(do=0, nb_layers=2, activation=ReLU):
    inputs = Input((None, None, 3))
    x = Conv2D(16, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               use_bias=False,
               kernel_regularizer=l2(1E-4))(inputs)

    conv1 = denseblock(x, 3, nb_layers, 32,
               dropout_rate=do, weight_decay=1E-4)
    pool1 = transition(conv1, 3, 64,
               dropout_rate=do, weight_decay=1E-4)

    conv2 = denseblock(pool1, 3, nb_layers, 64,
               dropout_rate=do, weight_decay=1E-4)
    pool2 = transition(conv2, 3, 128,
               dropout_rate=do, weight_decay=1E-4)

    conv3 = denseblock(pool2, 3, nb_layers, 128,
               dropout_rate=do, weight_decay=1E-4)
    
    up4 = concatenate([transition_transpose(conv3, 3, 64), conv2], axis=3)
    conv4 = denseblock(up4, 3, nb_layers, 64,
               dropout_rate=do, weight_decay=1E-4)
    
    up5 = concatenate([transition_transpose(conv4, 3, 32), conv1], axis=3)
    conv5 = activation()(denseblock(up5, 3, nb_layers, 32,
               dropout_rate=do, weight_decay=1E-4))
    
    conv6 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])

    model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy, metrics=['accuracy'])


    return model
 
 
def read_input(path):
    x = np.array(Image.open(path))/255.
    return x
 
 
def read_gt(path):
    x = np.array(Image.open(path))/255.
    return x[..., np.newaxis]
 
 
def random_crop(img, mask, crop_size=input_shape[0]):
    imgheight= img.shape[0]
    imgwidth = img.shape[1]
    i = randint(0, imgheight-crop_size)
    j = randint(0, imgwidth-crop_size)
 
    return img[i:(i+crop_size), j:(j+crop_size), :], mask[i:(i+crop_size), j:(j+crop_size)]
 
 
def gen(data, au=False):
    while True:
        repeat = 4
        index= random.choice(list(range(len(data))), batch_size//repeat)
        index = list(map(int, index))
        list_images_base = [read_input(data[i][0]) for i in index]
        list_gt_base = [read_gt(data[i][1]) for i in index]
 
        list_images = []
        list_gt = []
 
        for image, gt in zip(list_images_base, list_gt_base):
 
            for _ in range(repeat):
                image_, gt_ = random_crop(image.copy(), gt.copy())
                list_images.append(image_)
                list_gt.append(gt_)
 
        list_images_aug = []
        list_gt_aug = []
 
        for image, gt in zip(list_images, list_gt):
            if au:
                image, gt = random_augmentation(image, gt)
            list_images_aug.append(image)
            list_gt_aug.append(gt)
 
        yield np.array(list_images_aug), np.array(list_gt_aug)
 
 
if __name__ == '__main__':
 
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dropout", required=False,
                    help="dropout", type=float, default=0.2)
    ap.add_argument("-a", "--activation", required=False,
                    help="activation", default="ReLU")
    ap.add_argument("-n", "--nblayers", required=False,
                    help="nblayers", type=int, default=2)
 
    args = vars(ap.parse_args())
 
    activation = globals()[args['activation']]
 
    model_name = "DAU-Net_do_%s_nblayers_%s_activation_%s_"%(args['dropout'], args['nblayers'], args['activation'])
 
    print("Model : %s"%model_name)
 
    train_data = list(zip(sorted(glob('drive/MyDrive/DRIVE/training/images/*.tif')),
                          sorted(glob('drive/MyDrive/DRIVE/training/1st_manual/*.gif'))))
 
    model = get_daunet(do=args['dropout'], nb_layers=args['nblayers'], activation=activation)
 
    file_path = "drive/MyDrive/" + model_name + "weights.best.hdf5"
    #try:
    #    model.load_weights(file_path, by_name=True)
    #except:
    #    pass
 
 
 
 
    history = model.fit_generator(gen(train_data, au=True), epochs=100, verbose=2,
                         steps_per_epoch= 100*len(train_data)//batch_size)
 
    model.save_weights(file_path)