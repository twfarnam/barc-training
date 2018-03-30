import json
import os
from .generator import make_generator
from .database import categories
from shutil import rmtree
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from coremltools.converters.keras import convert


def train_inception(epochs=None, log_dir=None):
    batch_size = 16

    base_model = InceptionV3(
        input_shape=(299, 299, 3,),
        include_top=False,
        weights='imagenet',
    )

    category_ids, labels, class_weights = categories()

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(labels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    if os.path.exists(log_dir): rmtree(log_dir)
    os.makedirs(log_dir)

    tb_callback = TensorBoard(
        log_dir=log_dir,
        batch_size=batch_size,
        write_grads=True,
        write_images=True
    )

    train_generator, steps = make_generator(
        target_size=(299, 299, ),
        batch_size=batch_size,
        category_ids=category_ids,
    )

    model.fit_generator(
        train_generator(),
        steps_per_epoch=steps,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[ tb_callback ],
    )

    for layer in model.layers[:249]: layer.trainable = False
    for layer in model.layers[249:]: layer.trainable = True

    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy'
    )

    model.fit_generator(
        train_generator(),
        steps_per_epoch=steps,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[ tb_callback ]
    )

    if not os.path.exists('model'):
        os.makedirs('model')

    with open('model/labels.json', 'w') as fp:
        json.dump(labels, fp)

    model.save('model/inception.h5')

    # category_ids, labels = categories()
    # from keras.models import load_model
    # model = load_model('model/inception.h5')
    # print(model.inputs)

    coreml_model = convert(
        model,
        class_labels=labels,
        image_scale=2./255,
        red_bias=-1,
        green_bias=-1,
        blue_bias=-1,
    )

    coreml_model.save('model/inception.mlmodel')


