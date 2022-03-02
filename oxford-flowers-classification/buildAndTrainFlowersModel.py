# imports
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
from tensorflow import keras
from preprocessDefinition import preprocess

# retrieve dataset
train, traininfo = tfds.load("oxford_flowers102", split='train', as_supervised=True, with_info=True)
val, valinfo = tfds.load("oxford_flowers102", split='validation', as_supervised=True, with_info=True)
n_classes = traininfo.features["label"].num_classes # save class number for layer creation later

# reprocess and shuffle the dataset, add batching and preprocessing
batch_size = 32
train = train.shuffle(1000)
train = train.map(preprocess).batch(batch_size).prefetch(1)
val = val.map(preprocess).batch(batch_size).prefetch(1)

# load a VGG model pretrained on ImageNet and add a global average pooling layer and classification layer on top
base_model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

# freeze the base model layers to train the layers we just added
for layer in base_model.layers:
    layer.trainable = False

# add early stopping and checkpoint callbacks
checkpoint_cb=keras.callbacks.ModelCheckpoint('flowersModeltop.h5', save_best_only=True)
earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

# compile the model before training
optimizer = keras.optimizers.SGD(lr=0.5)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train, epochs=25, validation_data=val, callbacks=[checkpoint_cb,earlyStop_cb])

# load model with top layers trained
model = keras.models.load_model('flowersModeltop.h5')

# unfreeze layers
for layer in model.layers[-2:]:
  layer.trainable = True

# train some more
checkpoint_cb=keras.callbacks.ModelCheckpoint('flowersModel.h5', save_best_only=True)
optimizer = keras.optimizers.SGD(lr=3e-2)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train, epochs=30, validation_data=val, callbacks=[checkpoint_cb,earlyStop_cb])