#! /usr/local/bin/python

import os
import shutil
from subprocess import call

video_dir = 'videos'
image_dir = 'images'

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

for slug in os.listdir(video_dir):
    d = os.path.join(video_dir, slug)
    if not os.path.isdir(d): continue
    slug_dir = os.path.join(image_dir, slug)
    if not os.path.exists(slug_dir): os.makedirs(slug_dir)
    for f in os.listdir(d):
        if (f[-3:].lower() == 'mov'):
            out_dir = os.path.join(image_dir, slug, f)
            if os.path.exists(out_dir):
                print('exists: ' + f)
            else:
                print('processing: ' + f)
                os.makedirs(out_dir)
                call([
                    "ffmpeg",
                    "-i", os.path.join(d, f),
                    "-vf", "scale=84:150",
                    "-r", "5/1",
                    os.path.join(out_dir, "%03d.jpg"),
                ])



