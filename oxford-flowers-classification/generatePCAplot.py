import tensorflow.compat.v2 as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from preprocessDefinition import preprocess

# load a VGG model pretrained on ImageNet and add a global average pooling layer and classification layer on top
base_model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
model = keras.Model(inputs=base_model.input, outputs=avg)
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#model.summary()

# retrieve dataset
test, testinfo = tfds.load("oxford_flowers102", split='test', as_supervised=True, with_info=True)
n_classes = testinfo.features["label"].num_classes
n_examples = testinfo.splits["test"].num_examples

# temporary preprocess function
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.vgg16.preprocess_input(resized_image)
    return final_image, label

# get dataset
batch_size = 32
test = test.map(preprocess).batch(batch_size).prefetch(1)

# predict on data
y = model.predict(test)

# perform principal component analysis on data
pca = PCA(n_components = 12)
pca2dim = pca.fit_transform(y)
evr = pca.explained_variance_ratio_

# plot and save to image with matplotlib
plt.plot(evr, 'b*')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Number')
plt.savefig('explainedVariancePlot.png')
#plt.show()