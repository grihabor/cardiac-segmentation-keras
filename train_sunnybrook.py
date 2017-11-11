#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import izip, islice

import sunnybrook
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


def _filter_contour_files(files, contour_type):
    return fnmatch.filter(
        files,
        'IM-0001-*-'+contour_type+'contour-manual.txt'
    )


def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in _filter_contour_files(files, contour_type)
    ]

    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)

    print('Number of examples: {:d}'.format(len(contours)))
    contours = list(map(Contour, contours))
    
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

    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    sunnybrook.train(
        contour_type=sys.argv[1],
        crop_size=100,
    )


if __name__== '__main__':
    main()
