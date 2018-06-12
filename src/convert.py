import os
import json
import keras
import keras_applications

keras_applications.set_keras_submodules(
    backend=keras.backend,
    engine=keras.engine,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils)

from keras_applications.mobilenet_v2 import relu6
from coremltools.converters.keras import convert

def convert_model(architecture):
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

    labels_path = os.path.join(root, 'model/labels.json')
    with open(labels_path, 'r') as fp:
        labels = json.load(fp)

    if architecture == 'mobilenets':
        model_path = os.path.join(root, 'model/mobilenets.h5')
        model = keras.models.load_model(
                    model_path,
                    custom_objects={ 'relu6': relu6 })
    elif architecture == 'inception':
        model_path = os.path.join(root, 'model/inception.h5')
        model = keras.models.load_model(model_path)
    else:
        raise ValueError('architecture must be inception or mobilenets.')

    coreml_model = convert(
        model,
        input_names='image',
        image_input_names='image',
        class_labels=labels,
        image_scale=2./255,
        red_bias=-1,
        green_bias=-1,
        blue_bias=-1,
    )

    coreml_model.save("model/%s.mlmodel" % architecture)

