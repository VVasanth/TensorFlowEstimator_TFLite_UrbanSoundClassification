import tensorflow as tf
import numpy as np

from tensorflow import keras

ckpt_directory = "./urban_est_model_ckpt_dir/"

def get_network():
    num_filters = [24, 32, 64, 128]
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (60, 41, 2)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='features'))
    model.add(keras.layers.Conv2D(24, kernel_size,
                                  padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(128, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax", name="label"))

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model

model_ckpt = get_network()

latest = tf.train.latest_checkpoint(ckpt_directory)

model_ckpt.load_weights(latest)

est_model = tf.keras.estimator.model_to_estimator(keras_model=model_ckpt)

load_dir = "UrbanSound8K/processed/"

def getTestDataSet():
    test_index = 'fold9'
    x_test, y_test = [], []

    # read features or segments of an audio file
    test_data = np.load("{0}/{1}.npz".format(load_dir, test_index),
                         allow_pickle=True)
    # for training stack all the segments so that they are treated as an example/instance
    features = np.concatenate(test_data["features"], axis=0)
    x_test.append(features)
    # stack x,y pairs of all training folds
    x_test = np.concatenate(x_test, axis=0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x_test))
    print(dataset)
    dataset = dataset.map(map_features)
    return dataset.batch(100)


def map_features(features):
    """ Select features and annotation of the given sample. """
    input_ = {'features': features
              }
    return (input_)


# predict with the model and print results
pred_input_fn = getTestDataSet
pred_results = est_model.predict(input_fn=pred_input_fn)
print(next(pred_results))
print(next(pred_results))

print(100)


