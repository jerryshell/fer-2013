import tensorflow.keras as keras


def create_model_resnet_101():
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs'
    )

    resnet = keras.applications.ResNet101V2(
        include_top=False,
        weights=None,
        input_shape=(48, 48, 1),
        pooling='avg'
    )(inputs)

    outputs = keras.layers.Dense(
        units=7,
        activation='softmax',
        name='outputs',
    )(resnet)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model


def create_my_model():
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs'
    )

    x = keras.layers.SeparableConv2D(
        filters=64,
        kernel_size=3,
        name='cnn1',
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D(
        name='max_pool1',
    )(x)

    x = keras.layers.SeparableConv2D(
        filters=128,
        kernel_size=3,
        name='cnn2',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(name='relu2')(x)
    x = keras.layers.MaxPooling2D(
        name='max_pool2',
    )(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dropout(
        rate=0.5,
        name='dropout',
    )(x)

    outputs = keras.layers.Dense(
        units=7,
        activation='softmax',
        name='outputs',
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model


def create_model_gitshanks():
    # input
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs'
    )

    base_filters = 64

    # hidden 1
    x = keras.layers.Conv2D(
        filters=base_filters,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01),
    )(inputs)
    x = keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
    )(x)
    x = keras.layers.Dropout(
        rate=0.5,
    )(x)

    # hidden 2
    x = keras.layers.Conv2D(
        filters=2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        filters=2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
    )(x)
    x = keras.layers.Dropout(
        rate=0.5,
    )(x)

    # hidden 3
    x = keras.layers.Conv2D(
        filters=2 * 2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        filters=2 * 2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
    )(x)
    x = keras.layers.Dropout(
        rate=0.5,
    )(x)

    # hidden 4
    x = keras.layers.Conv2D(
        filters=2 * 2 * 2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        filters=2 * 2 * 2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
    )(x)
    x = keras.layers.Dropout(
        rate=0.5,
    )(x)

    # flatten
    x = keras.layers.Flatten()(x)

    # dense 1
    x = keras.layers.Dense(
        units=2 * 2 * 2 * base_filters,
        activation='relu',
    )(x)
    x = keras.layers.Dropout(
        rate=0.4
    )(x)

    # dense 2
    x = keras.layers.Dense(
        units=2 * 2 * base_filters,
        activation='relu',
    )(x)
    x = keras.layers.Dropout(
        rate=0.4
    )(x)

    # dense 3
    x = keras.layers.Dense(
        units=2 * base_filters,
        activation='relu',
    )(x)
    x = keras.layers.Dropout(
        rate=0.5
    )(x)

    # output
    outputs = keras.layers.Dense(
        units=7,
        activation='softmax',
        name='outputs',
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model


def test_resnet_101():
    model = create_model_resnet_101()
    model.summary()


def test_my_model():
    model = create_my_model()
    model.summary()


def test_model_gitshanks():
    model = create_model_gitshanks()
    model.summary()


if __name__ == '__main__':
    test_resnet_101()
    test_my_model()
    test_model_gitshanks()
