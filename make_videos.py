#! /usr/local/bin/python

import os
import glob
import shutil
from subprocess import call

training_dir = '/Users/omar/code/barc/training'
video_dir = os.path.join(training_dir, 'videos')
image_dir = os.path.join(training_dir,'images')
data_dir = os.path.join(training_dir,'data')

# if os.path.exists(image_dir):
#     shutil.rmtree(image_dir)
# os.makedirs(image_dir)

# # Make images out of videos

# for f in os.listdir(video_dir):
#     if (f[-3:] == 'MOV'):
#         print(f[:-4])
#         out_dir = os.path.join(image_dir, f[:-4])
#         os.makedirs(out_dir)
#         call([
#             "ffmpeg",
#             "-i", os.path.join(video_dir, f),
#             "-r", "1/1",
#             os.path.join(out_dir, "%03d.jpg"),
#         ])

# # Create data directory

# if os.path.exists(data_dir):
#     shutil.rmtree(data_dir)
# os.makedirs(data_dir)

for d in os.listdir(image_dir):
    for i, f in enumerate(os.listdir(os.path.join(image_dir, d))):
        src_path = os.path.join(image_dir, d, f)
        dest = 'validation' if i % 5 == 0 else 'train' 
        dest_dir = os.path.join(data_dir, d, dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_path = os.path.join(dest_dir, f)
        # print(src_path, dest_path)
        shutil.copyfile(src_path, dest_path)


