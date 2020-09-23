import tensorflow.keras as keras
from tensorflow.python.keras.applications import resnet


def stack_fn(x):
    x = resnet.stack2(x, 64, 2, stride1=1, name='conv2')
    x = resnet.stack2(x, 128, 2, stride1=2, name='conv3')
    x = resnet.stack2(x, 256, 2, stride1=2, name='conv4')
    x = resnet.stack2(x, 512, 2, stride1=2, name='conv5')
    return x


def create_model_resnet_50():
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
        name='inputs'
    )

    x = resnet.ResNet(
        stack_fn=stack_fn,
        preact=False,
        use_bias=False,
        model_name='resnet18',
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(48, 48, 1),
        pooling='avg',
        classes=1000,
        classifier_activation='softmax'
    )(inputs)

    # resnet = keras.applications.ResNet50V2(
    #     include_top=False,
    #     weights=None,
    #     input_shape=(48, 48, 1),
    #     pooling='avg'
    # )(inputs)

    x = keras.layers.Dropout(
        rate=0.5
    )(x)

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
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    return model


def create_my_model_64():
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


def test_resnet_50():
    model = create_model_resnet_50()
    model.summary()


def test_resnet_101():
    model = create_model_resnet_101()
    model.summary()


def test_my_model_64():
    model = create_my_model_64()
    model.summary()


if __name__ == '__main__':
    test_resnet_50()
    test_resnet_101()
    test_my_model_64()
