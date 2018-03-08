#! /usr/local/bin/python

import json
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.callbacks import TensorBoard
from coremltools.converters.keras import convert

batch_size = 16
epochs = 1
n_samples = 0
n_test_samples = 0

for slug in os.listdir('data/train'):
    n_samples += len(os.listdir(os.path.join('data/train', slug)))

for slug in os.listdir('data/validation'):
    n_test_samples += len(os.listdir(os.path.join('data/train', slug)))

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=batch_size
)

# save the labels to labels.json
labels_dict = train_generator.class_indices
labels = [None] * len(labels_dict.items())
for key, value in labels_dict.iteritems():
    labels[value] = key
if not os.path.exists('model'):
    os.makedirs('model')
with open('model/labels.json', 'w') as fp:
    json.dump(labels, fp)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=batch_size
)

# load the model
base_model = MobileNet(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    classes=len(labels)
)

# make the top of the model, using our number of classes
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Reshape(shape, name='reshape_1')(x)
x = Dropout(dropout, name='dropout')(x)
x = Conv2D(len(labels), (1, 1), padding='same', name='conv_preds')(x)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((len(labels),), name='reshape_2')(x)
model = Model(inputs=base_model.input, outputs=x)

# Train top layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./log',
    write_graph=True,
    write_images=True
)

# train the top of the model
model.fit_generator(
    train_generator,
    steps_per_epoch= n_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= n_test_samples // batch_size
    callbacks=[tbCallBack]
)



# Fine tune lower layers

# print layer names
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# train only the top 2 blocks
for layer in model.layers[:70]: layer.trainable = False
for layer in model.layers[70:]: layer.trainable = True

# recompile the model for these modifications to take effect
model.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    loss='categorical_crossentropy'
)

model.fit_generator(
    train_generator,
    steps_per_epoch= n_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= n_test_samples // batch_size,
    callbacks=[tbCallBack]
)

model.save('model/mobilenets.h5')

# XXX must add output labels here
coreml_model = convert(
    model,
    input_names='image',
    image_input_names='image',
    output_names=labels,
)

coreml_model.save('model/mobilenets.mlmodel')


