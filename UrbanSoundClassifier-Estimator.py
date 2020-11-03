from pathlib import Path

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

def get_network():
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


def _create_estimator():
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    model = get_network()

    est_model = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="urban_est_model_ckpt_dir", config=tf.estimator.RunConfig(
        save_checkpoints_steps=300,
        tf_random_seed=3,
        save_summary_steps=5,
        session_config=session_config,
        log_step_count_steps=10,
        keep_checkpoint_max=2
    ))
    return est_model

def _create_train_spec():
    """ Creates train spec.

    :param params: TF params to build spec from.
    :returns: Built train spec.
    """
    input_fn = getTrainingDataSet

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=1000)
    return train_spec



load_dir = "UrbanSound8K/processed/"
folds = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                  'fold5', 'fold6', 'fold7', 'fold8',
                  'fold9', 'fold10'])
inpFeatures = None

def getTrainingDataSet():
    train_index= range(1,9)
    x_train, y_train = [], []
    for ind in train_index:
        # read features or segments of an audio file
        train_data = np.load("{0}/{1}.npz".format(load_dir, folds[ind]),
                             allow_pickle=True)
        # for training stack all the segments so that they are treated as an example/instance
        features = np.concatenate(train_data["features"], axis=0)
        labels = np.concatenate(train_data["labels"], axis=0)
        x_train.append(features)
        y_train.append(labels)
    # stack x,y pairs of all training folds
    x_train = np.concatenate(x_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    print(dataset)
    dataset = dataset.map(map_features)
    dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(32)


def getEvalDataSet():
    global inpFeatures
    test_index = 'fold10'
    x_test, y_test = [], []

    # read features or segments of an audio file
    test_data = np.load("{0}/{1}.npz".format(load_dir, test_index),
                         allow_pickle=True)
    # for training stack all the segments so that they are treated as an example/instance
    features = np.concatenate(test_data["features"], axis=0)
    labels = np.concatenate(test_data["labels"], axis=0)
    x_test.append(features)
    y_test.append(labels)
    # stack x,y pairs of all training folds
    x_test = np.concatenate(x_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    print(dataset)
    dataset = dataset.map(map_features)
    inpFeatures = dataset.element_spec[0]['features']
    return dataset.batch(100)



def map_features(features, label):
        """ Select features and annotation of the given sample. """
        input_ = {'features':features
           }
        output = {'label':label
           }
        return (input_, output)


def _create_evaluation_spec():
    """ Setup eval spec evaluating ever n seconds

    :param params: TF params to build spec from.
    :returns: Built evaluation spec.
    """
    input_fn = getEvalDataSet
    evaluation_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,
        throttle_secs=600)
    return evaluation_spec


estimator = _create_estimator()
train_spec = _create_train_spec()
evaluation_spec = _create_evaluation_spec()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
eval_result = tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        evaluation_spec)
print('Eval result: {}'.format(eval_result))


def serving_input_fn():
    inputs = {'features': tf.compat.v1.placeholder(dtype=tf.float32,shape=[1,60,41,2], name='features')}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


export_dir = './urban_export_savedmodel_dir/'

estimator.export_saved_model(export_dir_base=export_dir, serving_input_receiver_fn=serving_input_fn)

subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])

converter = tf.lite.TFLiteConverter.from_saved_model(latest, signature_keys=['serving_default'])
converter.allow_custom_ops = True
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.uint8, tf.float32]
tflite_model = converter.convert()

with open('./tflite/model.tflite', 'wb') as f:
  f.write(tflite_model)

print(100)
