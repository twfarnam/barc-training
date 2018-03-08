#! /usr/bin/env python

import os
from shutil import copyfile, rmtree

image_dir = 'images'
data_dir = 'data'

if os.path.exists(data_dir):
    rmtree(data_dir)
os.makedirs(data_dir)

for slug in os.listdir(image_dir):
    images = []
    for vid in os.listdir(os.path.join(image_dir, slug)):
        for f in os.listdir(os.path.join(image_dir, slug, vid)):
            images.append(os.path.join(vid, f))
    for i, f in enumerate(images):
        src_path = os.path.join(image_dir, slug, f)
        dest = 'validation' if i % 5 == 0 else 'train' 
        dest_dir = os.path.join(data_dir, dest, slug)
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)
        dest_path = os.path.join(dest_dir, str(i) + '.jpg')
        # print(src_path, dest_path)
        copyfile(src_path, dest_path)

