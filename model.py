import keras
from keras import Input
from keras.layers import Lambda, ZeroPadding2D, BatchNormalization, Conv2D, MaxPooling2D

from fcn_model import dice_coef_loss, mvn


class ConvBlock:
    def __init__(self, filters, index):
        self.filters = filters
        self.index = index

    def __call__(self, x):
        x = Conv2D(
            kernel_size=3,
            filters=self.filters,
            name='conv1_{}'.format(self.index),
            activation='relu',
        )(x)
        x = BatchNormalization(
            name='batchnorm1_{}'.format(self.index),
        )(x)

        x = Conv2D(
            kernel_size=3,
            filters=self.filters,
            name='conv2_{}'.format(self.index),
            activation='relu',
        )(x)
        x = BatchNormalization(
            name='batchnorm2_{}'.format(self.index),
        )(x)

        return x


def fcn_model(input_shape, num_classes, weights=None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )

    x = Input(shape=input_shape, dtype='float', name='data')
    x = BatchNormalization(mvn, name='batchnorm0')(x)
    x = ZeroPadding2D(padding=5, name='pad0')(x)

    conv1 = ConvBlock(filters=64, index=1)(x)
    pool1 = MaxPooling2D(padding='valid', name='pool1')

    conv


    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                         padding='valid', name='pool1')(mvn3)

    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2,
                         padding='valid', name='pool2')(mvn7)

    conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2,
                         padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)

    conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)

    score_conv15 = Conv2D(filters=num_classes, kernel_size=1,
                          strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True,
                          name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                                strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False,
                                name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1,
                          strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True,
                          name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                                strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False,
                                name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=None, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True,
                         name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                                strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False,
                                name='upsample3')(fuse_scores2)
    crop3 = Lambda(crop, name='crop3')([data, upsample3])
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True,
                         name='predictions')(crop3)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss,
                  metrics=['accuracy', dice_coef, jaccard_coef])

    return model

