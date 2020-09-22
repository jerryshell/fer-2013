import tensorflow.keras as keras

from data_helper import DataHelper


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


if __name__ == '__main__':
    print('data loading...')
    data_helper = DataHelper()

    model = create_model_resnet_101()
    model.summary()
