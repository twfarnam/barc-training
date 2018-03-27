import json
import os
from .generator import make_generator
from .database import categories
from shutil import rmtree
from keras.models import Model
from keras.layers import Reshape, GlobalAveragePooling2D, Dropout, Conv2D, Activation
from keras.optimizers import SGD
from keras.applications.mobilenet import MobileNet
from keras.callbacks import TensorBoard
from coremltools.converters.keras import convert


def train_mobilenets(epochs=None, log_dir=None):
    batch_size = 16

    base_model = MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )

    category_ids, labels = categories()

    # make the top of the model, using our number of classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(len(labels), (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((len(labels),), name='reshape_2')(x)
    model = Model(inputs=base_model.input, outputs=x)

    # train the top
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    if os.path.exists(log_dir): rmtree(log_dir)
    os.makedirs(log_dir)

    # tb_callback = TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=1,
    #     batch_size=batch_size,
    #     write_grads=True,
    #     write_graph=True,
    #     write_images=True
    # )

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
        # callbacks=[ tb_callback ],
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
        # callbacks=[ tb_callback ],
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
    )

    coreml_model.save('model/mobilenets.mlmodel')


