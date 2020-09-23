import glob
import os

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

IMAGE_SIZE = 256
CHANNEL = 3
test_dir = "C:/Users/zenkori/Documents/ObjectTracking/data/val/"
train_dir = "C:/Users/zenkori/Documents/ObjectTracking/data/train/"
print(test_dir)
input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL))# https://qiita.com/haru1977/items
 # model define
x = Conv2D(28, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(28, (3, 3), activation='relu',  padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(CHANNEL, (3, 3), padding='same')(x)
#
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()
#
# # データの読み込み/17833e508fe07c004119
x_train = []
for picture in glob.glob(os.path.join(train_dir, '*.jpg')):
    img = img_to_array(load_img(picture, target_size=(IMAGE_SIZE, IMAGE_SIZE), grayscale=False))
    x_train.append(img)


x_test = []
for picture in glob.glob(os.path.join(test_dir, '*.jpg')):
    img = img_to_array(load_img(picture, target_size=(IMAGE_SIZE, IMAGE_SIZE), grayscale=False))
    x_test.append(img)

train_num = len(x_train)
test_num = len(x_test)
print('train num = ', train_num)
print('test num = ', test_num)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.reshape(x_train, (train_num, IMAGE_SIZE, IMAGE_SIZE, CHANNEL))
x_test = np.reshape(x_test, (test_num, IMAGE_SIZE, IMAGE_SIZE, CHANNEL))

history = autoencoder.fit(x_train, x_train,
                epochs=3000,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                )

model_json = autoencoder.to_json()
with open('ae_pinapple.json', 'w') as json_file:
    json_file.write(model_json)
autoencoder.save_weights('ae_weights.h5')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()