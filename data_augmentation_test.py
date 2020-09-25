import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from data_helper import DataHelper

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)


def data_augmentation_fn():
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
    )

    x = inputs

    x = keras.layers.experimental.preprocessing.Rescaling(1. / 255)(x)

    x = keras.layers.experimental.preprocessing.RandomFlip('horizontal')(x)
    x = keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


data_helper = DataHelper()

train_data_generator = data_helper.train_data_generator(1)

data, label = next(train_data_generator)

daf = data_augmentation_fn()

plt.figure(figsize=(10, 10))
for i in range(9):
    # augmented_images = data_augmentation(data)
    augmented_images = daf(data)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy(), cmap='gray')
    plt.axis("off")
plt.show()
