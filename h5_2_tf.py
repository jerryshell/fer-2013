from tensorflow import keras

model = keras.models.load_model(filepath='./model/fer.66.99.h5')
model.save('./model_tf', save_format='tf')
