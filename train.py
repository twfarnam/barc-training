#! /usr/local/bin/python

import json
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3, preprocess_input

batch_size = 16
epochs = 20
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
    target_size=(299, 299),
    batch_size=batch_size
)

labels_dict = train_generator.class_indices
labels = [None] * len(labels_dict.items())
for key, value in labels_dict.iteritems():
    labels[value] = key
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
    target_size=(299, 299),
    batch_size=batch_size
)



if not os.path.exists('model'):
    os.makedirs('model')

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(labels), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model = multi_gpu_model(model, gpus=4)


# Train top layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(
    train_generator,
    steps_per_epoch= n_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= n_test_samples // batch_size
)



# Fine tune all layers

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]: layer.trainable = False
for layer in model.layers[249:]: layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./log',
    write_graph=True,
    write_images=True
)

model.fit_generator(
    train_generator,
    steps_per_epoch= n_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= n_test_samples // batch_size,
    callbacks=[tbCallBack]
)

model.save('model/nn.h5')


