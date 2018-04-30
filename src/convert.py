import os
import json
import keras
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from coremltools.converters.keras import convert

def convert_model(architecture):
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

    labels_path = os.path.join(root, 'model/labels.json')
    with open(labels_path, 'r') as fp:
        labels = json.load(fp)

    if architecture == 'mobilenets':
        model_path = os.path.join(root, 'model/mobilenets.h5')
        layers = { 'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D }
        with CustomObjectScope(layers):
            model = keras.models.load_model(model_path)
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

