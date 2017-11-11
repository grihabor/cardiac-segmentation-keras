import re
import fnmatch
import os
import numpy as np
import cv2
import dicom

from helpers import center_crop


def _shrink_case(case):
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
        self.case = _shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return '<Contour for case {}, image {}>'.format(self.case, self.img_no)

    def __repr__(self):
        return str(self)


def _filter_contour_files(files, contour_type):
    return fnmatch.filter(
        files,
        'IM-0001-*-' + contour_type + 'contour-manual.txt'
    )


def load_all_contours(contour_path, contour_type, shuffle=True):
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


def export_all_contours(contours, data_path, crop_size, sax_series):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path, sax_series)
        images[idx] = center_crop(img, crop_size=crop_size)
        masks[idx] = center_crop(mask, crop_size=crop_size)

    return images, masks


def read_contour(contour, data_path, sax_series):
    filename = 'IM-%s-%04d.dcm' % (sax_series[contour.case], contour.img_no)
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
