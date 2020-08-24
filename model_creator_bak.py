import tensorflow.keras as keras

from data_helper import DataHelper


def create_model_cnn():
    inputs = keras.layers.Input(
        shape=(48 * 48,),
        name='inputs'
    )

    reshape = keras.layers.Reshape(
        target_shape=(48, 48),
        name='reshape'
    )(inputs)

    cnn1 = keras.layers.SeparableConv1D(
        filters=64,
        kernel_size=3,
        name='cnn1',
    )(reshape)
    bn1 = keras.layers.BatchNormalization()(cnn1)
    relu1 = keras.layers.ReLU(name='relu1')(bn1)
    max_pool1 = keras.layers.MaxPooling1D(
        name='max_pool1',
    )(relu1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=128,
        kernel_size=3,
        name='cnn2',
    )(max_pool1)
    bn1 = keras.layers.BatchNormalization()(cnn2)
    relu2 = keras.layers.ReLU(name='relu2')(bn1)
    max_pool2 = keras.layers.MaxPooling1D(
        name='max_pool2',
    )(relu2)

    flatten = keras.layers.Flatten()(max_pool2)

    # dropout = keras.layers.Dropout(rate=0.5)(flatten)

    outputs = keras.layers.Dense(
        units=7,
        activation='softmax',
        name='outputs',
    )(flatten)

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
