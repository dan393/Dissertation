import tensorboard as tensorboard
import seaborn as seaborn
from tensorflow.python.client import device_lib
import tensorflow as tf

print('tensorflow' + tf.__version__)
print('tensorboard' + tensorboard.__version__)
print('seaborn' + seaborn.__version__)
tf.config.list_physical_devices('GPU')
tf.test.is_built_with_cuda
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')
device_lib.list_local_devices()

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

data_dir = "../input/pets"
test_path = os.path.join(data_dir, 'test')
train_path = os.path.join(data_dir, 'train')
os.listdir(train_path)

cat_image = train_path + '/cats/' + os.listdir(train_path + '/cats')[0]
imread(cat_image).shape
# plt.imshow(imread(cat_image))

dog_image = train_path + '/dogs/' + os.listdir(train_path + '/dogs/')[5]
plt.imshow(imread(dog_image))
dog_image

len(os.listdir(train_path + '/dogs'))

dim1 = []
dim2 = []

for image_filename in os.listdir(test_path + '/dogs/'):
    img = imread(test_path + '/dogs/' + image_filename)
    #     print (train_path+'\\dogs\\' + image_filename)
    d1, d2, _ = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(dim1, dim2)

np.mean(dim2)
imread(dog_image).shape
image_shape = (200, 200, 3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rescale=1 / 255, fill_mode='nearest')
# image_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, rescale =1/255,
#                                    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')


# plt.imshow(imread(cat_image))
cat_image
plt.imshow(image_gen.random_transform(imread(cat_image)))

image_gen.flow_from_directory(train_path)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

import keras_resnet.models
import keras

x = keras.layers.Input(image_shape)
model = keras_resnet.models.ResNet50(x, classes=1)

# model = Sequential()

# model.add(Conv2D(filters=128, kernel_size=(4,4), input_shape=image_shape, activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Conv2D(filters=256, kernel_size=(4,4), input_shape=image_shape, activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Conv2D(filters=256, kernel_size=(4,4), input_shape=image_shape, activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Flatten())

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)
batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2],
                                                color_mode='rgb', batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path, target_size=image_shape[:2],
                                               color_mode='rgb', batch_size=batch_size,
                                               class_mode='binary', shuffle=False)

train_image_gen.class_indices

help(model.fit)

print ("starting model training:")

results = model.fit(train_image_gen, epochs=10, verbose=1, validation_data=test_image_gen, callbacks=[early_stop],
                    use_multiprocessing=True)

history = results.history
history

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])

plt.plot(history['loss'])
plt.plot(history['val_loss'])
model.evaluate_generator(test_image_gen)

pred = model.predict_generator(test_image_gen)

predictions = pred > 0.5

pred
len(pred)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(test_image_gen.classes, predictions))

confusion_matrix(test_image_gen.classes, predictions)

dog_image

from tensorflow.keras.preprocessing import image

my_image = image.load_img(dog_image, target_size=image_shape)
my_image

my_img_arr = image.img_to_array(my_image)
my_img_arr.shape

my_img_arr = np.expand_dims(my_img_arr, axis=0)
my_img_arr.shape

model.predict(my_img_arr)

get_ipython().system(' pip install keras-resnet')
