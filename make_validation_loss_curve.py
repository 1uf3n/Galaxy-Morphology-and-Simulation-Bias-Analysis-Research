# Imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

## Metrics and data
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Others
import time
import zipfile
import itertools

print("Imports Complete")

df_valid = pd.read_csv('df_valid_plus_petroRad_r.csv')

class CustomDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, directory,
                x_col, y_col, batch_size,
                input_size=(424,424,3), shuffle=True):

        self.dataframe = dataframe.copy()
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.dataframe.index)

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __get_image(self, path, target_size):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize_with_crop_or_pad(image_arr, 207, 207)

        return image_arr/255.0

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        # Generate data containing batch_size samples

        filename_batch = batches[self.x_col['filename']]
        alfa_batch = batches[self.x_col['alfa']]
        label_batch = batches[self.y_col]

        x0_batch = np.asarray([self.__get_image(self.directory+filename, self.input_size) for filename in filename_batch])
        x1_batch = alfa_batch.values
        y_batch = label_batch.values

        return x0_batch, x1_batch, y_batch

    def __getitem__(self, index):
        batches = self.dataframe[index * self.batch_size:(index + 1) * self.batch_size]
        x0, x1, y = self.__get_data(batches)
        return x0, x1, y

    def __len__(self):
        return self.n // self.batch_size

# Model
base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(207, 207, 3),
                      pooling=None)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

resnet_50 = keras.Model(inputs=base_model.input, outputs=output)

valid_acc_metric = keras.metrics.SparseCategoricalAccuracy()
valid_loss_metric = keras.metrics.SparseCategoricalCrossentropy()

theta = 0.01896268 # 11.25 normalized
def parametric_bias(alfa, theta):
    alfa = tf.cast(alfa, dtype=tf.float64)
    theta = tf.cast(theta, dtype=tf.float64)

    alfa_square = tf.math.square(alfa)
    theta_square = tf.math.square(theta)
    theta_square = tf.math.scalar_mul(2, theta_square)
    tmp = tf.math.divide(alfa_square, theta_square)
    tmp = tf.math.scalar_mul(-1, tmp)
    result = tf.math.exp(tmp)
    return result

def custom_loss_fn(p, alfa, label):
    theta = tf.cast(0.01896268, dtype=tf.float64)
    label = tf.cast(label, dtype=tf.float64)
    one = tf.cast(1.0, dtype=tf.float64)

    # p_i
    p = tf.cast(p, dtype=tf.float64)
    # s = tf.nn.softmax(p, axis=1)
    s0 = tf.gather(p, 0, axis=1)
    s1 = tf.gather(p, 1, axis=1)
    s0 = tf.transpose(s0)
    s1 = tf.transpose(s1)

    # p_0|1
    pb = parametric_bias(alfa, theta)

    # a_i
    a = tf.math.log(s1)
    a = tf.math.multiply(label, a)
    a = tf.math.multiply(a, pb)

    # b_i
    b = tf.math.log(s0)
    b = tf.math.multiply((one - label),  b)

    epsilon = tf.cast(1e-5, dtype=tf.float64)
    res = tf.math.add(a, b)
    res = tf.math.scalar_mul(-1, res)
    res = tf.math.add(epsilon, res)
    loss = tf.math.reduce_sum(res)
    return loss

batch_size = 32
input_shape=(207, 207, 3)
images_directory = '/scratch/christoq_root/christoq0/jjfisch/'

valid_generator = CustomDataGenerator(dataframe=df_valid, directory=images_directory,
                                      x_col={'filename':'iauname', 'alfa':'petroRad_r_psf'},
                                      y_col='label', batch_size=batch_size, input_size=input_shape)

valid_dataset = tf.data.Dataset.from_generator(lambda:valid_generator,
                                               output_signature=(
                                               tf.TensorSpec(shape=(batch_size, 207, 207, 3), dtype=tf.float32),
                                               tf.TensorSpec(shape=(batch_size,), dtype=tf.float64),
                                               tf.TensorSpec(shape=(batch_size,), dtype=tf.int64)))

valid_dataset = valid_dataset.prefetch(64)

@tf.function
def test_step(x, y, alfa):
    valid_logits = resnet_50(x, training=False)
    loss_value = custom_loss_fn(valid_logits, alfa, y)
    valid_acc_metric.update_state(y, valid_logits)
    valid_loss_metric.update_state(y, valid_logits)

    return loss_value

valid_loss_epoch = []
valid_acc_epoch = []
valid_custom_loss_epoch = []

valid_min_loss = np.inf
valid_max_acc = -np.inf

weights_path = './weights_debiasing/'

valid_steps_per_epoch = valid_generator.n//valid_generator.batch_size

epochs = 9
for epoch in range(epochs):

    # Load weights
    resnet_50.load_weights(weights_path + f'epoch_{epoch}.h5')

    print('\nEpoch %d' % (epoch,))
    start_time = time.time()
    valid_loss_ = 0.0

    stp = 0
    for valid_step, (x0_batch_valid, x1_batch_valid, y_batch_valid) in enumerate(valid_dataset):
        print(valid_step)
        valid_loss_ += test_step(x0_batch_valid, y_batch_valid, x1_batch_valid)
        if valid_step == (valid_steps_per_epoch-1):
            stp = valid_step
            break

    valid_acc = valid_acc_metric.result()
    valid_loss = valid_loss_metric.result()
    valid_loss = valid_loss/(stp+1)
    valid_loss_ = valid_loss_/(stp+1)

    valid_acc_epoch.append(valid_acc)
    valid_loss_epoch.append(valid_loss)
    valid_custom_loss_epoch.append(valid_loss_)
    print("Validation acc: %.4f" % (float(valid_acc),))
    print("Validation custom loss: %.4f" % (float(valid_loss_),))

    print("Time taken: %.2fs" % (time.time() - start_time))

    print()
