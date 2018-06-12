import json
import os
import numpy
from datetime import datetime
from shutil import rmtree

import keras
import keras_applications

keras_applications.set_keras_submodules(
    backend=keras.backend,
    engine=keras.engine,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils)

from keras_applications.mobilenet_v2 import MobileNetV2 
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from coremltools.converters.keras import convert

from .generator import make_generator
from .database import categories


def train_mobilenets(epochs=None):

    batch_size = 16

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )

    category_ids, labels, class_weights = categories()

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(len(labels), activation='sigmoid', name='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)

    # train the top
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    timestamp = datetime.now().isoformat(' ')[:19]
    log_dir = os.path.join('log', timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tb_callback = TensorBoard(
        log_dir=log_dir,
        batch_size=batch_size,
        write_grads=True,
        write_images=True
    )

    train_generator, steps = make_generator(
        target_size=(224, 224, ),
        batch_size=batch_size,
        category_ids=category_ids,
    )

    # train the top of the model
    model.fit_generator(
        train_generator(),
        steps_per_epoch=steps,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[ tb_callback ],
    )

    # fine tune lower layers, only the top 2 blocks
    for layer in model.layers[:70]: layer.trainable = False
    for layer in model.layers[70:]: layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy'
    )

    model.fit_generator(
        train_generator(),
        steps_per_epoch=steps,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[ tb_callback ],
    )

    if not os.path.exists('model'):
        os.makedirs('model')

    with open('model/labels.json', 'w') as fp:
        json.dump(labels, fp)

    model.save('model/mobilenets.h5')

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

    coreml_model.save('model/mobilenets.mlmodel')


