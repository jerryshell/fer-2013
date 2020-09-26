from tensorflow import keras

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
])


def create_resnet():
    # inputs
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs'
    )
    x = inputs

    # data augmentation
    x = data_augmentation(x)

    # rescaling
    x = keras.layers.experimental.preprocessing.Rescaling(1. / 255)(x)

    # resnet
    x = keras.applications.ResNet50(
        weights=None,
        input_shape=(48, 48, 1),
        pooling='avg',
        classes=7,
    )(x)
    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    return model


def create_model_66():
    # input
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs',
    )
    x = inputs

    # # data augmentation
    # x = data_augmentation(x)
    #
    # # rescaling
    # x = keras.layers.experimental.preprocessing.Rescaling(1. / 255)(x)

    # hidden layers
    for filters in [64, 128, 256, 512]:
        x = keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=7,
            padding='same',
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=7,
            padding='same',
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.AveragePooling2D(
            pool_size=2,
            padding='same',
        )(x)
        x = keras.layers.Dropout(rate=0.5)(x)

    # hidden 2
    x = keras.layers.SeparableConv2D(
        filters=256,
        kernel_size=3,
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)

    # hidden 3
    x = keras.layers.SeparableConv2D(
        filters=7,
        kernel_size=3,
        padding='same',
    )(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # output
    x = keras.layers.Activation(activation='softmax')(x)
    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    return model


def create_model_64():
    # input
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs'
    )

    base_filters = 64

    # hidden 1
    x = keras.layers.SeparableConv2D(
        filters=base_filters,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(),
    )(inputs)
    x = keras.layers.SeparableConv2D(
        filters=base_filters,
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
    x = keras.layers.SeparableConv2D(
        filters=2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(
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
    x = keras.layers.SeparableConv2D(
        filters=2 * 2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(
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
    x = keras.layers.SeparableConv2D(
        filters=2 * 2 * 2 * base_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SeparableConv2D(
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
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    return model


def test_resnet():
    model = create_resnet()
    model.summary()


def test_model_64():
    model = create_model_64()
    model.summary()


def test_model_66():
    model = create_model_66()
    model.summary()


if __name__ == '__main__':
    test_resnet()
    test_model_64()
    test_model_66()
