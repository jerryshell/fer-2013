import matplotlib.pyplot as plt
from tensorflow import keras

from data_helper import DataHelper

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
])


def data_augmentation_fn():
    inputs = keras.layers.Input(
        shape=(48, 48, 1),
    )

    x = inputs

    # x = keras.layers.experimental.preprocessing.RandomTranslation(
    #     height_factor=0.1,
    #     width_factor=0.1,
    # )(x)

    x = keras.layers.experimental.preprocessing.RandomFlip('horizontal')(x)
    x = keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)
    x = keras.layers.experimental.preprocessing.RandomZoom(0.1)(x)

    x = keras.layers.experimental.preprocessing.Rescaling(1. / 255)(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


data_helper = DataHelper()

train_data_generator = data_helper.train_data_generator(1)

data, label = next(train_data_generator)

class_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print('label', label)
print('class', class_list[int(label)])

daf = data_augmentation_fn()
daf.summary()

plt.figure(figsize=(10, 10))
for i in range(9):
    # augmented_images = data_augmentation(data)
    augmented_images = daf(data)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy(), cmap='gray')
    plt.axis("off")
plt.show()
