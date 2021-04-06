# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install requests Pillow tqdm cmake dlib

# run in the workspace folder
# RUN curl --remote-name http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# RUN bzip2 -dv shape_predictor_68_face_landmarks.dat.bz2
