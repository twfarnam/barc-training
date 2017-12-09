#! /usr/local/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import time
import json


print 'start load'

with open('out/labels.json', 'r') as fp:
    labels = json.load(fp)

model = load_model('out/nn.h5')

img = load_img('data/validation/bedroom_desk/0.jpg', target_size=(224, 224))
data = img_to_array(img)
data = data.reshape((1,) + data.shape)

time1 = time.time()
prediction = model.predict(data)
time2 = time.time()

print(prediction)

print('prediction: ' + labels[prediction.argmax(axis=-1)[0]])

print 'prediction took %0.3f ms' % ((time2-time1)*1000.0)


