def preprocess(image, label):
    import tensorflow.compat.v2 as tf
    from tensorflow import keras
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    cropped_image = tf.image.random_crop(image, [min_dim, min_dim, 3])
    cropped_image = tf.image.random_flip_left_right(cropped_image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = keras.applications.vgg16.preprocess_input(resized_image)
    return final_image, label