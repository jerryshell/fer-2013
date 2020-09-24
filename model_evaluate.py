import tensorflow.keras as keras

from data_helper import DataHelper

batch_size = 64

print('data loading...')
data_helper = DataHelper()
train_data_generator = data_helper.train_data_generator(batch_size)

model = keras.models.load_model('fer.66.16.h5')

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['acc'],
)

model.evaluate(
    x=train_data_generator,
    steps=data_helper.train_data_count // batch_size,
)
