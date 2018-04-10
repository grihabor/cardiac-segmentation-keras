from keras.models import Model
from keras.layers import Input, Conv2D,\
        Conv2DTranspose, MaxPool2D,\
        Concatenate
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.losses import categorical_crossentropy

from math import sqrt
import json
import logging


model_log = logging.getLogger(__name__)

def weighted_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    if weights:
        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss
    else:
        return categorical_crossentropy

    
def dice_metric_2d():
    def _dice_metric_2d(y_true, y_pred, smooth=0.0):
        """Average dice coefficient per batch."""
        axes = (1, 2, 3)
        intersection = K.sum(y_true * y_pred, axis=axes)
        summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)

        return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)
    return _dice_metric_2d


def dice_loss_2d():
    def _dice_loss_2d(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)
    return _dice_loss_2d


def get_model(
        nx:int, ny:int, 
        chanel_num:int, class_num:int):
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_tensor = Input(shape=(nx, ny, chanel_num))
    input_weights_tensor = Input(shape=(nx, ny, class_num))
    conv_1_1 = Conv2D(64, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * chanel_num)), None))(input_tensor)
    conv_1_2 = Conv2D(64, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 64)), None))(conv_1_1)
    pool_1 = MaxPool2D((2, 2), (2, 2), padding='same')(conv_1_2)

    conv_1_3 = Conv2D(64, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 64)), None))(conv_1_2)
    conv_1_4 = Conv2D(64, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 64)), None))(conv_1_3)

    conv_2_1 = Conv2D(128, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 64)), None))(pool_1)
    conv_2_2 = Conv2D(128, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(conv_2_1)
    pool_2 = MaxPool2D((2, 2), (2, 2), padding='same')(conv_2_2)

    conv_2_3 = Conv2D(128, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(conv_2_2)
    conv_2_4 = Conv2D(128, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(conv_2_3)

    conv_3_1 = Conv2D(256, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(pool_2)
    conv_3_2 = Conv2D(256, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(conv_3_1)
    pool_3 = MaxPool2D((2, 2), (2, 2), padding='same')(conv_3_2)

    conv_3_3 = Conv2D(256, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(conv_3_2)
    conv_3_4 = Conv2D(256, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(conv_3_3)

    conv_4_1 = Conv2D(512, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(pool_3)
    conv_4_2 = Conv2D(512, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(conv_4_1)
    pool_4 = MaxPool2D((2, 2), (2, 2), padding='same')(conv_4_2)

    conv_4_3 = Conv2D(512, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(conv_4_2)
    conv_4_4 = Conv2D(512, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(conv_4_3)

    conv_5_1 = Conv2D(1024, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(pool_4)
    conv_5_2 = Conv2D(1024, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 1024)), None))(conv_5_1)

    deconv_4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 1024)), None))(conv_5_2)

    Model(inputs=[input_tensor], outputs=[deconv_4]).summary()
    return
    
    concat_4 = Concatenate(axis=3)([deconv_4, conv_4_4])
    up_conv_4_1 = Conv2D(512, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 1024)), None))(concat_4)
    up_conv_4_2 = Conv2D(512, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(up_conv_4_1)

    deconv_3 = Conv2DTranspose(256, (2, 2), (2, 2), padding='same', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(up_conv_4_2)
    concat_3 = Concatenate(axis=3)([deconv_3, conv_3_4])  
    up_conv_3_1 = Conv2D(256, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 512)), None))(concat_3)
    up_conv_3_2 = Conv2D(256, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(up_conv_3_1)
  
    deconv_2 = Conv2DTranspose(128, (2, 2), (2, 2), padding='same', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(up_conv_3_2)
    concat_2 = Concatenate(axis=3)([deconv_2, conv_2_4])    
    up_conv_2_1 = Conv2D(128, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 256)), None))(concat_2)
    up_conv_2_2 = Conv2D(128, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(up_conv_2_1)

    deconv_1 = Conv2DTranspose(64, (2, 2), (2, 2), padding='same', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(up_conv_2_2)
    concat_1 = Concatenate(axis=3)([deconv_1, conv_1_4])
    up_conv_1_1 = Conv2D(64, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 128)), None))(concat_1)
    up_conv_1_2 = Conv2D(64, 3, padding='same', activation='relu', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 64)), None))(up_conv_1_1)   
    up_conv_1_3 = Conv2D(class_num, 1, padding='same', activation='softmax', 
            kernel_initializer=RandomNormal(0.0, sqrt(2 / (27 * 64)), None))(up_conv_1_2)   

    if config['use_custom_weights']:
        inputs=[input_tensor, input_weights_tensor]
    else:
        inputs=input_tensor

    model = Model(
            inputs=inputs, outputs=up_conv_1_3)

    if 'adam' == config['optimizer']['name'].lower():
        optimizer = Adam(
                lr = float(config['optimizer']['learning_rate']))
    elif 'momentum' == config['optimizer']['name'].lower():
        optimizer = SGD(
                lr = config['optimizer']['learning_rate'],
                momentum=config['optimizer']['momentum'],
                decay=config['optimizer']['decay'],
                nesterov=True)
    else:
        model_log.error('Unknown optimizer in config')
        return None

    if 'dice_2d' == config['loss']['name']:
        loss = dice_loss_2d()
    elif 'weighted_crossentropy' == config['loss']['name']:
        if config['use_custom_weights']:
            loss = weighted_crossentropy(input_weights_tensor)
        else:
            loss = weighted_crossentropy()
    else:
        model_log.error('Unknown loss in config')
        return None

    model.compile(
            optimizer=optimizer, loss=loss,
            metrics=['accuracy', dice_metric_2d()])

    return model
