#! /usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
import time
import json
import random

print 'start load'

with open('model/labels.json', 'r') as fp:
    labels = json.load(fp)

model = load_model('model/inception.h5')
model._make_predict_function()

for slug in labels:
    directory = os.path.join('data', 'validation', slug)
    images = os.listdir(directory)
    image_path = os.path.join(directory, random.choice(images))
    image = load_img(image_path, target_size=(224, 224))
    data = img_to_array(image)
    data = data.reshape((1,) + data.shape)
    data = preprocess_input(data)

    time1 = time.time()
    prediction = model.predict(data)
    time2 = time.time()

    index = prediction.argmax(axis=-1)[0]
    label = labels[index]
    confidence = prediction[0][index] * 100
    duration = ((time2-time1)*1000.0)

    if slug == label:
        data =  (label, confidence, duration)
        print 'CORRECT: %s with %0.0f%% in %0.0f ms' % data
    else:
        data =  (label, slug, confidence, duration)
        print 'WRONG: guessed "%s" for "%s" with %0.0f%% in %0.0f ms' % data

