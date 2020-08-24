import tensorflow.keras as keras

from data_helper import DataHelper


def create_model_cnn():
    inputs = keras.layers.Input(
        shape=(48 * 48,),
        name='inputs'
    )

    reshape = keras.layers.Reshape(
        target_shape=(48, 48, 1),
        name='reshape'
    )(inputs)

    resnet = keras.applications.resnet.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(48, 48, 1),
        pooling='max'
    )(reshape)

    # flatten = keras.layers.Flatten()(resnet)

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


if __name__ == '__main__':
    print('data loading...')
    data_helper = DataHelper()

    model = create_model_cnn()
    model.summary()
