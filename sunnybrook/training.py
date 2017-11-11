

def train(*, contour_type, crop_size):
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

