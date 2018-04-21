import numpy
from random import shuffle
from .database import images, categories, cat_for_image
from PIL import Image as PILImage
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import Sequence


def one_image_as_data(image_id, target_size, datagen):
    path = './images/%s.jpg' % image_id
    img = load_img(path, target_size=target_size)
    x = img_to_array(img)
    x = datagen.random_transform(x.astype('float32'))
    x = datagen.standardize(x)
    return preprocess_input(x, mode='tf')

# target_size:
#   (224, 224,) for mobilenets
#   (299, 299,) for inception
def make_generator(target_size=None, batch_size=None, category_ids=None):

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    image_ids = images()
    steps = len(image_ids) // batch_size

    def train_generator():
        while 1:
            shuffle(image_ids)
            for i in range(steps):
                start = i * batch_size
                end = (i + 1) * batch_size
                ids = image_ids[start:end]
                x = numpy.array([
                    one_image_as_data(id, target_size, datagen) for id in ids
                ])
                y = numpy.array([
                    cat_for_image(id, category_ids) for id in ids
                ])
                yield (x, y)

    return (train_generator, steps, )
    # return (train_generator, 1, )


