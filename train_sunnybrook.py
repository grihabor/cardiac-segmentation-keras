#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import izip, islice

from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES

SEED = 1234

EPOCH_COUNT = 40
BATCH_SIZE = 1

np.random.seed(SEED)

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = os.environ['SUNNYBROOK_ROOT_PATH']

TRAIN_CONTOUR_PATH = os.path.join(
    SUNNYBROOK_ROOT_PATH,
    'Sunnybrook_Cardiac_MR_Database_ContoursPart3',
    'Sunnybrook Cardiac MR Database ContoursPart3',
    'TrainingDataContours'
)
TRAIN_IMG_PATH = os.path.join(
    SUNNYBROOK_ROOT_PATH,
    'challenge_training'
)


def shrink_case(case):
    toks = case.split('-')
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return '-'.join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)
    
    __repr__ = __str__


def read_contour(contour, data_path):
    filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(data_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask


def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours


def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks


def main():

    if len(sys.argv) < 3:
        sys.exit('Usage: python {} <i/o> <gpu_id>'.format(sys.argv[0]))

    contour_type = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 100

    print('Mapping ground truth {} contours to images in train...'.format(contour_type))
    train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True)
    print('Done mapping training set')
    
    split = int(0.1 * len(train_ctrs))

    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    
    print()
    print('Building Train dataset ...')
    img_train, mask_train = export_all_contours(
        train_ctrs,
        TRAIN_IMG_PATH,
        crop_size=crop_size,
    )

    print()
    print('Building Dev dataset ...')
    img_dev, mask_dev = export_all_contours(
        dev_ctrs,
        TRAIN_IMG_PATH,
        crop_size=crop_size,
    )
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    model = fcn_model(input_shape, num_classes, weights=None)
    
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
        img_train,
        shuffle=False,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    mask_generator = mask_datagen.flow(
        mask_train,
        shuffle=False,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    train_generator = izip(image_generator, mask_generator)
    
    max_iter = (len(train_ctrs) / BATCH_SIZE) * EPOCH_COUNT
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    learning_rate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)

    for epoch in range(1, EPOCH_COUNT + 1):
        print()
        print('Main Epoch {:d}'.format(epoch))
        print('Learning rate: {:6f}'.format(learning_rate))
        train_result = []

        iter_count = len(img_train) // BATCH_SIZE
        for img, mask in islice(train_generator, iter_count):
            res = model.train_on_batch(img, mask)
            curr_iter += 1
            learning_rate = lr_poly_decay(model, base_lr, curr_iter,
                                          max_iter, power=0.5)
            train_result.append(res)

        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)

        print('Train result {:s}:'.format(model.metrics_names))
        print('{:s}'.format(train_result))
        print()
        print('Evaluating dev set ...')

        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)

        print()
        print('Dev set result {:s}:'.format(model.metrics_names))
        print('{:s}'.format(result))

        save_file = '_'.join([
            'sunnybrook',
            contour_type,
            'epoch',
            str(epoch),
        ]) + '.h5'

        os.makedirs('model_logs', exist_ok=True)
        save_path = os.path.join('model_logs', save_file)

        print()
        print('Saving model weights to {:s}'.format(save_path))

        model.save_weights(save_path)


if __name__== '__main__':
    main()
