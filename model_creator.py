from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
])


def create_xception(input_shape=(48, 48, 1), num_classes=7):
    inputs = keras.Input(shape=input_shape)

    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    return model


def create_res_net():
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


def create_mobile_net():
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
    x = keras.applications.MobileNet(
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
    filters_list = [64, 128, 256, 512]
    kernel_size_list = [7, 5, 3, 3]
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        print('filters', filters, 'kernel_size', kernel_size)
        x = keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
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


def test_xception():
    model = create_xception()
    model.summary()
    keras.utils.plot_model(model, to_file='xception.png', show_shapes=True)


def test_res_net():
    model = create_res_net()
    model.summary()
    keras.utils.plot_model(model, to_file='res_net.png', show_shapes=True)


def test_mobile_net():
    model = create_mobile_net()
    model.summary()
    keras.utils.plot_model(model, to_file='mobile_net.png', show_shapes=True)


def test_model_64():
    model = create_model_64()
    model.summary()
    keras.utils.plot_model(model, to_file='model_64.png', show_shapes=True)


def test_model_66():
    model = create_model_66()
    model.summary()
    keras.utils.plot_model(model, to_file='model_66.png', show_shapes=True)


if __name__ == '__main__':
    test_xception()
    test_res_net()
    test_mobile_net()
    test_model_64()
    test_model_66()
