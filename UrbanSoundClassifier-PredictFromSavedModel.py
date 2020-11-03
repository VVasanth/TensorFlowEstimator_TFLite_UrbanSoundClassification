from pathlib import Path

import tensorflow as tf
import numpy as np

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
    return dataset.batch(10)


def map_features(features):
    """ Select features and annotation of the given sample. """
    input_ = {'features': features
              }
    return (input_)


export_dir = './urban_export_savedmodel_dir/'
subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])


predict_fn = tf.saved_model.load(latest)
inference_func = predict_fn.signatures["serving_default"]

for batch in getTestDataSet().take(1):
    preds = inference_func(batch['features'])['label']
    topPred = tf.argmax(preds, 1).numpy()
    print(topPred)

print(100)