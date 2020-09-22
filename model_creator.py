import tensorflow.keras as keras


def create_model_resnet_101():
    inputs = keras.layers.Input(
        shape=(48, 48),
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
        shape=(48, 48),
        name='inputs'
    )

    x = keras.layers.SeparableConv1D(
        filters=64,
        kernel_size=3,
        name='cnn1',
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling1D(
        name='max_pool1',
    )(x)

    x = keras.layers.SeparableConv1D(
        filters=128,
        kernel_size=3,
        name='cnn2',
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(name='relu2')(x)
    x = keras.layers.MaxPooling1D(
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


def test_resnet_101():
    model = create_model_resnet_101()
    model.summary()


def test_my_model():
    model = create_my_model()
    model.summary()


if __name__ == '__main__':
    # test_resnet_101()
    test_my_model()
