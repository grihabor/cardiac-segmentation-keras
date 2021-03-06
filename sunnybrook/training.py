import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import islice

from fcn_model import fcn_model
from gridnet_model import gridnet_model

from helpers import lr_poly_decay, get_SAX_SERIES

from .contour import load_all_contours, export_all_contours
from skimage import io, color

DIR_DATA = './data'


def _to_image(output):
    arr = np.asarray(output)
    normalized = arr / np.max(arr)
    return (
        normalized
        if normalized.shape[-1] != 1
        else np.reshape(normalized, normalized.shape[:-1])
    )


def predict_and_save(model, images, *, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictions = model.predict(images)
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        prediction = prediction[:, :, 1]
        
        image_path = os.path.join(output_dir, 'img_{}.png'.format(i))
        prediction_path = os.path.join(output_dir, 'pred_{}.png'.format(i))

        io.imsave(image_path, _to_image(image))
        io.imsave(prediction_path, _to_image(prediction))


def train(image_path, contour_path, *, contour_type, crop_size, batch_size, seed, epoch_count):
    contours = load_all_contours(contour_path, contour_type, shuffle=True)

    loaded_train_x, loaded_train_y = export_all_contours(
        contours,
        image_path,
        crop_size=crop_size,
        sax_series=get_SAX_SERIES(),
    )

    input_shape = (crop_size, crop_size, 1)
    num_classes = 2

    m = fcn_model

    model = m(input_shape, num_classes)

    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    image_generator = image_datagen.flow(
        loaded_train_x,
        shuffle=False,
        batch_size=batch_size,
        seed=seed,
    )
    mask_generator = mask_datagen.flow(
        loaded_train_y,
        shuffle=False,
        batch_size=batch_size,
        seed=seed,
    )
    train_generator = zip(image_generator, mask_generator)

    max_iter = (len(contours) / batch_size) * epoch_count
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    learning_rate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)

    for epoch in range(1, epoch_count + 1):
        print()
        print('Main Epoch {:d}'.format(epoch))
        print('Learning rate: {:6f}'.format(learning_rate))
        train_result = []

        iter_count = len(img_train) // batch_size
        for img, mask in islice(train_generator, iter_count):
            res = model.train_on_batch(img, mask)
            curr_iter += 1
            learning_rate = lr_poly_decay(model, base_lr, curr_iter,
                                          max_iter, power=0.5)
            train_result.append(res)

        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)

        print('Train result {}:'.format(str(model.metrics_names)))
        print('{}'.format(str(train_result)))
        print()
        print('Evaluating dev set ...')

        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)

        print()
        print('Dev set result {}:'.format(str(model.metrics_names)))
        print('{}'.format(str(result)))

        predict_and_save(
            model,
            img_dev[:10],
            output_dir=os.path.join(DIR_DATA, 'predictions'),
        )

        save_file = '_'.join([
            'sunnybrook',
            contour_type,
            'epoch',
            str(epoch),
        ]) + '.h5'

        logs_dir = os.path.join(DIR_DATA, 'model_logs')
        os.makedirs(logs_dir, exist_ok=True)
        save_path = os.path.join(logs_dir, save_file)

        print()
        print('Saving model weights to {}'.format(save_path))

        model.save_weights(save_path)

