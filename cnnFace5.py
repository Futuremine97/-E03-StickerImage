#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:19:58 2022

@author: parksunghun
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import tensorflow as tf
import sys

import dlib
import argparse
import time
print(os.getenv('HOME'))
my_image_path = os.getenv('HOME')+'/Downloads/e0e0.png'
img_bgr = cv2.imread(my_image_path)    # OpenCV로 이미지를 불러옵니다
my_image_path2 = os.getenv('HOME')+'/Downloads/56555.png'
img_bgr2 = cv2.imread(my_image_path2) 
   # OpenCV로 이미지를 불러옵니다

t1 = img_bgr.reshape(-1,)
t2 = t1[:8033280]
t3= t2.reshape(1046,2560,3)
# plt.imshow(img_bgr)
# handle command line arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to image file')
# ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
#                 help='path to weights file')
# args = ap.parse_args()
# detector_faster_Rcnn = hub.load(module_handle1).signatures['default'] #detector에 사용할 모듈 저장
# detector_ssd = hub.load(module_handle2).signatures['default'] #detector에 사용할 모듈 저장
# run_detector(detector_faster_Rcnn, my_image_path)

# apply face detection (cnn)
img2 = dlib.load_rgb_image(my_image_path)
img22 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = dlib.load_rgb_image(my_image_path2)
img33 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
t1 = img2.reshape(-1,1)
t2 = t1[:6903600]
t3= t2.reshape(1046,2200,3)

cnn_face_detector = dlib.cnn_face_detection_model_v1(os.getenv('HOME')+'/Downloads/mmod_human_face_detector.dat')#가중치 파일 제공해야 함!!!
faces_cnn = cnn_face_detector(img22, 1)


# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
     # draw box over face
    cv2.rectangle(img22, (x,y), (x+w,y+h), (0,0,255), 2)


# write at the top left corner of the image
# for color identification
img_height, img_width = img22.shape[:2]

cv2.putText(img22, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

# display output image
cv2.imshow("face detection with dlib.png", img22)
cv2.imwrite("face detection with dlib.png", img22)

print(t3.shape)
print(img3.shape)
print(img2.shape)

##interpolation
width1 = 2200
height1 = 1040
img_stack_sm2 = np.zeros((height1, width1, 3), dtype = int)

for idx in range(len(t3)):
    img8 = img2[idx, :, :]
    img_sm8 = cv2.resize(img8, (width1, height1), interpolation=cv2.INTER_CUBIC)
    img_stack_sm2[idx, :, :] = img_sm8
##

##interpolation
width2 = 2200
height2 = 1040
img_stack_sm3 = np.zeros((height2, width2, 3),dtype = int)

for idx in range(len(t3)):
    img9 = img3[idx, :, :]
    img_sm9 = cv2.resize(img9, (width2, height2), interpolation=cv2.INTER_CUBIC)
    img_stack_sm3[idx, :, :] = img_sm9
##

t4 = img3.reshape(-1,)
t5 = t4[:8033280]
t6= t5.reshape(1046,2560,3)
print(t6.shape)
# close all windows
t3t = np.array(t3)
t6t = np.array(t6)

print(img_stack_sm2.shape)
print(img_stack_sm3.shape)

e1 = cv2.addWeighted(img_stack_sm2, 0.5 , img_stack_sm3 , 0.7, 2.4)
cv2.imshow('e1', e1)
cv2.imwrite("addweighted.png",e1)
