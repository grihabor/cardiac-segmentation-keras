#!/usr/bin/env python2.7

import os
import sys
import numpy as np
import sunnybrook

SEED = 1234

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


def main():

    if len(sys.argv) < 3:
        sys.exit('Usage: python {} <i/o> <gpu_id>'.format(sys.argv[0]))

    np.random.seed(SEED)

    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    sunnybrook.train(
        TRAIN_IMG_PATH,
        TRAIN_CONTOUR_PATH,
        contour_type=sys.argv[1],
        crop_size=100,
        batch_size=1,
        seed=SEED,
        epoch_count=40,
    )


if __name__== '__main__':
    main()
