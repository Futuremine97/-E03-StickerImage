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
my_image_path = os.getenv('HOME')+'/Downloads/8080.png'
img_bgr = cv2.imread(my_image_path)    # OpenCV로 이미지를 불러옵니다
my_image_path2 = os.getenv('HOME')+'/Downloads/4545.png'
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
print("img22.shape")
print(img22.shape)
print("img33.shape")
print(img33.shape)
img_height, img_width = img22.shape[:2]

cv2.putText(img22, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

# display output image
cv2.imshow("face detection with dlib3.png", img22)
cv2.imwrite("face detection with dlib3.png", img22)

print(t3.shape)
print(img3.shape)
print(img2.shape)

tq1 = np.resize(img2, (1040,2200,3))
tq2 = np.resize(img3, (1040,2200,3))
##interpolation
#width1 = 2200
#height1 = 1040

#img_stack_sm2 = np.zeros((height1, width1, 1040), dtype = int)
#for idx in range(len(tq1)-1):
#    img8 = tq1[idx, :, :]
#    img_sm8 = cv2.resize(img8, (height1, width1), interpolation=cv2.INTER_CUBIC)
#    img_stack_sm2[idx, :, :] = img_sm8
##
##interpolation
width1= 900
height1 = 900
#img_stack_sm6 = np.zeros((width1, height1, 1040),dtype = int)
img_sm10 = cv2.resize(img22, (height1, width1), interpolation=cv2.INTER_CUBIC)#보간법. cubic 이 linear 보다 좋음.
#img_stack_sm6[idx, :, :] = img_sm10
##


##interpolation
#width2 = 2200
#height2 = 1040
#img_stack_sm3 = np.zeros((height2, width2, 1040),dtype = int)
#print("len(tq2)-1")
#print(len(tq2)-1)
#for idx in range(len(tq2)-1):
#    img9 = tq2[idx, :, :]
#    img_sm9 = cv2.resize(img9, (height2, width2), interpolation=cv2.INTER_CUBIC)#보간법. cubic 이 linear 보다 좋음.
#    img_stack_sm3[idx, :, :] = img_sm9
##

##interpolation
width2 = 900
height2 = 900
#img_stack_sm5 = np.zeros((width1, height1, 1040),dtype = int)
img_sm9 = cv2.resize(img33, (height2, width2), interpolation=cv2.INTER_CUBIC)#보간법. cubic 이 linear 보다 좋음.
#img_stack_sm5[idx, :, :] = img_sm9
##

#t4 = img3.reshape(-1,)
#t5 = t4[:8033280]
#t6= t5.reshape(1046,2560,3)
#print(t6.shape)

# close all windows
#t3t = np.array(t3)
#t6t = np.array(t6)
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[30]) # 코의 index는 30 입니다
    x = landmark[30][0]
    y = landmark[30][1] - dlib_rect.height()//2
    w = h = dlib_rect.width()
img_sticker = cv.resize(img_sm9,(90,90))
refined_x = x - 40 // 2
refined_y = y - 40

if refined_x < 0: 
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
if refined_y < 0:
    img_sticker = img_sticker[-refined_y:, :]
    refined_y = 0

sticker_area = img_sm10[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_sm10[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.show()

#print(img_stack_sm2.shape)
#print(img_stack_sm3.shape)
#t_q1 = np.array(img_stack_sm2)
#t_q2 = np.array(img_stack_sm3)

#tqqq1 = np.resize(img_stack_sm2,(1040,2200,3))
#tqqq2 = np.resize(img_stack_sm3,(1040,2200,3))

#e1 = cv2.addWeighted(img_sm10, 0.5 , img_sm9 , 0.7, 2.4)
#cv2.imshow('e1', e1)
#cv2.imwrite("addweighted3.png",e1)



