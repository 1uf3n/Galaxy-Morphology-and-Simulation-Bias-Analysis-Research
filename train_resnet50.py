import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt

# ---------- Configuration ----------
TRAIN_CSV   = os.environ.get("TRAIN_CSV_OVERRIDE", "df_train.csv")
VALID_CSV   = os.environ.get("VALID_CSV_OVERRIDE", "df_valid.csv")
IMAGE_DIR   = os.environ.get("IMAGE_DIR_OVERRIDE", "/path/to/images/")
BATCH_SIZE  = 32
IMG_SIZE    = (424, 424)
EPOCHS      = 10
NUM_CLASSES = 2
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ---------- Build filename→path mapping (recursive) ----------
all_pngs = list(Path(IMAGE_DIR).rglob("*.png"))
fname2path = {p.name: str(p) for p in all_pngs}

# ---------- Dataframe loading ----------
def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # make sure we have a basename column
    if 'filename' not in df.columns:
        if 'iauname' in df.columns:
            df['filename'] = df['iauname'].apply(lambda x: os.path.basename(x))
        else:
            raise KeyError("CSV must contain 'iauname' or 'filename'")

    # map into fullpaths via our recursive scan
    df['filepath'] = df['filename'].map(fname2path)

    # drop any missing files
    df = df.dropna(subset=['filepath','label']).reset_index(drop=True)
    return df

train_df = load_dataframe(TRAIN_CSV)
valid_df = load_dataframe(VALID_CSV)

# ---------- tf.data pipelines ----------
def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    img = preprocess_input(img)
    return img, label

def make_dataset(df, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values,
                                             df['label'].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, shuffle=True)
valid_ds = make_dataset(valid_df, shuffle=False)

# ---------- Model definition ----------
base_model = ResNet50(include_top=False, weights=None,
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- Callbacks ----------
checkpoint_cb = ModelCheckpoint(
    os.path.join(WEIGHTS_DIR, "best.weights.h5"),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True
)
csv_logger = CSVLogger("training.log")

# ---------- Training ----------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[checkpoint_cb, csv_logger]
)

# ---------- Plot & Save ----------
plt.figure()
plt.plot(history.history['loss'],     label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig('learning_curve.pdf'); plt.clf()

plt.figure()
plt.plot(history.history['accuracy'],     label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.savefig('accuracy_curve.pdf')

# # Imports

# ## Model
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.applications.resnet import ResNet50
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# ## Metrics and data
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ## Standard imports
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# ## Others
# import time
# import zipfile
# import itertools

# # Load data and train, valid, test separation
# # path = 'csv_for_paper/'
# # data = pd.read_csv(path +'data_for_nn.csv')
# # df_train = pd.read_csv(path +'df_train_for_paper.csv')
# # df_valid = pd.read_csv(path +'df_valid_for_paper.csv')
# # df_test = pd.read_csv(path +'df_test_for_paper.csv')

# # df_train = pd.read_csv('df_train.csv')
# # df_valid = pd.read_csv('df_valid.csv')

# import os
# train_csv = os.environ.get("TRAIN_CSV_OVERRIDE", "df_train.csv")
# valid_csv = os.environ.get("VALID_CSV_OVERRIDE", "df_valid.csv")
# df_train = pd.read_csv(train_csv)
# df_valid = pd.read_csv(valid_csv)

# # Custom data generator
# ## Normalize and crop (424x424 -> 424x424)
# class CustomDataGeneratorIMG(tf.keras.utils.Sequence):

#     def __init__(self, dataframe, directory,
#                 x_col, y_col, batch_size,
#                 input_size=(424,424,3), shuffle=True):

#         self.dataframe = dataframe.copy()
#         self.directory = directory
#         self.x_col = x_col
#         self.y_col = y_col
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.shuffle = shuffle

#         self.n = len(self.dataframe.index)

#     def on_epoch_end(self):
#         if self.shuffle:
#             self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

#     # def __get_image(self, path, target_size):
#     #     image = tf.keras.preprocessing.image.load_img(path)
#     #     image_arr = tf.keras.preprocessing.image.img_to_array(image)
#     #     image_arr = tf.image.resize_with_crop_or_pad(image_arr, 424, 424)
#     #     return image_arr/255.0

#     def __get_image(self, filename, target_size):
#         image_dir = os.environ.get("IMAGE_DIR_OVERRIDE", "/default/path/")
#         path = os.path.join(image_dir, filename)
#         print(f"Loading image from: {path}")
#         image = tf.keras.preprocessing.image.load_img(path)
#         image_arr = tf.keras.preprocessing.image.img_to_array(image)
#         image_arr = tf.image.resize_with_crop_or_pad(image_arr, 424, 424)
#         return image_arr / 255.0

#     def __get_output(self, label, num_classes):
#         return tf.keras.utils.to_categorical(label, num_classes=num_classes)

#     def __get_data(self, batches):
#         # Generate data containing batch_size samples

#         filename_batch = batches[self.x_col['iauname']]
#         label_batch = batches[self.y_col]

#         x0_batch = np.asarray([self.__get_image(self.directory+filename, self.input_size) for filename in filename_batch])
#         y_batch = label_batch.values

#         return x0_batch, y_batch

#     def __getitem__(self, index):
#         batches = self.dataframe[index * self.batch_size:(index + 1) * self.batch_size]
#         x0, y = self.__get_data(batches)
#         return x0, y

#     def __len__(self):
#         return self.n // self.batch_size


# # ResNet50
# base_model = ResNet50(include_top=False,
#                       weights=None,
#                       input_shape=(424, 424, 3),
#                       pooling=None)

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# output = Dense(2, activation='softmax')(x)

# resnet_50 = keras.Model(inputs=base_model.input, outputs=output)

# # Optimizer and loss function
# optimizer = Adam(learning_rate=1e-4)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# # Prepare metrics
# train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
# valid_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# train_loss_metric = keras.metrics.SparseCategoricalCrossentropy()
# valid_loss_metric = keras.metrics.SparseCategoricalCrossentropy()

# # TF datasets
# batch_size = 32
# input_shape=(424, 424, 3)

# # images_directory = '/scratch/christoq_root/christoq0/jjfisch/'
# images_directory = os.environ.get("IMAGE_DIR_OVERRIDE", "/scratch/christoq_root/christoq0/jjfisch/")

# ## Custom generators
# train_generator = CustomDataGeneratorIMG(dataframe=df_train, directory=images_directory,
#                                       x_col={'iauname':'iauname', 'alfa':'petroRad_r_psf'},
#                                       y_col='label', batch_size=batch_size, input_size=input_shape)

# valid_generator = CustomDataGeneratorIMG(dataframe=df_valid, directory=images_directory,
#                                       x_col={'iauname':'iauname', 'alfa':'petroRad_r_psf'},
#                                       y_col='label', batch_size=batch_size, input_size=input_shape)

# ## Tensorflow datasets
# train_dataset = tf.data.Dataset.from_generator(lambda:train_generator,
#                                                output_signature=(
#                                                tf.TensorSpec(shape=(32, 424, 424, 3), dtype=tf.float32),
#                                                tf.TensorSpec(shape=(32,), dtype=tf.int64)))


# train_dataset = train_dataset.prefetch(64)

# valid_dataset = tf.data.Dataset.from_generator(lambda:valid_generator,
#                                                output_signature=(
#                                                tf.TensorSpec(shape=(32, 424, 424, 3), dtype=tf.float32),
#                                                tf.TensorSpec(shape=(32,), dtype=tf.int64)))


# valid_dataset = valid_dataset.prefetch(64)

# # Train and test steps
# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = resnet_50(x, training=True)
#         loss_value = loss_fn(y, logits)
#         loss_value += sum(resnet_50.losses)

#     grads = tape.gradient(loss_value, resnet_50.trainable_weights)
#     optimizer.apply_gradients(zip(grads, resnet_50.trainable_weights))
#     train_acc_metric.update_state(y, logits)
#     train_loss_metric.update_state(y, logits)
#     return loss_value

# @tf.function
# def test_step(x, y):
#     valid_logits = resnet_50(x, training=False)
#     loss_value = loss_fn(y, valid_logits)
#     valid_acc_metric.update_state(y, valid_logits)
#     valid_loss_metric.update_state(y, valid_logits)

#     return loss_value

# # Training loop

# ## Auxiliar variables
# train_loss_epoch = []
# train_acc_epoch = []
# train_custom_loss_epoch = []

# valid_loss_epoch = []
# valid_acc_epoch = []
# valid_custom_loss_epoch = []

# valid_min_loss = np.inf
# valid_max_acc = -np.inf

# weights_path = './weights/'

# ## Training loop
# train_steps_per_epoch = train_generator.n//train_generator.batch_size
# valid_steps_per_epoch = valid_generator.n//valid_generator.batch_size

# epochs = 10
# for epoch in range(0, epochs):
#     print('\nEpoch %d' % (epoch,))
#     start_time = time.time()
#     train_loss_ = 0.0
#     valid_loss_ = 0.0

#     ### Iterate over the batches of the dataset
#     stp = 0
#     for step, (x0_batch_train, y_batch_train) in enumerate(train_dataset):
#         loss_value = train_step(x0_batch_train, y_batch_train)
#         train_loss_ += loss_value

#         # Log every 50 batches.
#         if step % 100 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                  % (step, float(loss_value))
#             )
#             print("Seen so far: %s samples" % ((step + 1) * train_generator.batch_size))

#         if step == (train_steps_per_epoch-1):
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                  % (step, float(loss_value))
#             )
#             print("Seen so far: %s samples" % ((step + 1) * train_generator.batch_size))
#             stp = step
#             break

#     ## Display metrics at the end of each epoch.
#     train_acc = train_acc_metric.result()
#     train_loss = train_loss_metric.result()
#     train_acc_epoch.append(train_acc)
#     train_loss_epoch.append(train_loss/(stp+1))
#     train_custom_loss_epoch.append(train_loss_/(stp+1))

#     print("Training acc over epoch: %.4f" % (float(train_acc),))
#     print("Training custom loss over epoch: %.4f" % (float(train_loss_/(stp+1)),))

#     ## Reset training metrics at the end of each epoch
#     train_acc_metric.reset_states()
#     train_loss_metric.reset_states()
#     train_loss_ = 0.0

#     print("Time taken: %.2fs" % (time.time() - start_time))

#     # Save weights from last epoch
#     save_path = weights_path + 'last_epoch.h5'
#     resnet_50.save_weights(save_path)
#     print('Weights saved to ' + save_path)


#     # Run a validation loop at the end of each epoch.
#     stp = 0
#     for valid_step, (x0_batch_valid, y_batch_valid) in enumerate(valid_dataset):
#         valid_loss_ += test_step(x0_batch_valid, y_batch_valid)
#         if valid_step == (valid_steps_per_epoch-1):
#             stp = valid_step
#             break

#     valid_acc = valid_acc_metric.result()
#     valid_loss = valid_loss_metric.result()
#     valid_loss = valid_loss/(stp+1)
#     valid_loss_ = valid_loss_/(stp+1)

#     valid_acc_epoch.append(valid_acc)
#     valid_loss_epoch.append(valid_loss)
#     valid_custom_loss_epoch.append(valid_loss_)
#     print("Validation acc: %.4f" % (float(valid_acc),))
#     #print("Validation loss: %.4f" % (float(valid_loss),))
#     print("Validation custom loss: %.4f" % (float(valid_loss_),))

#     #print("Time taken: %.2fs" % (time.time() - start_time))

#     print()

#     # Save weights if valid_loss improved
#     if valid_loss_ < valid_min_loss:
#         print('Validation loss improved from %.4f to %.4f, saving model weights' % (float(valid_min_loss), float(valid_loss_)))
#         resnet_50.save_weights(weights_path + 'weights_loss.h5')
#         valid_min_loss = valid_loss_
#     else:
#         print('Validation loss did not improve from %.4f' % (float(valid_min_loss),))


#     valid_acc_metric.reset_states()
#     valid_loss_metric.reset_states()
#     valid_loss_ =  0.0
#     print("Time taken: %.2fs" % (time.time() - start_time))

# # Learning curve
# plt.plot(train_loss_epoch, label='Train')
# plt.plot(valid_loss_epoch, label = 'Validation')
# plt.title('Learning Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='lower left')
# plt.savefig('learning_curve.pdf')
# plt.clf()

# # Accuracy
# plt.plot(train_acc_epoch, label='Train')
# plt.plot(valid_acc_epoch, label = 'Validation')
# plt.title('Accuracy Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.savefig('accuracy_curve.pdf')
