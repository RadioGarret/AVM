#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
import MK as mk
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np

import glob
import time

# 위처럼 python3에서 opencv와 ros 충돌을 막으려면 import cv2 하기전에 "if~" 부분 적어야한다. 

####################
##
## 참고사항
##
####################
# 1. 아래의 소스코드는, 정지된 이미지 4장을 정합하여 AVM 만드는 과정을 나타낸 것으로, 
#    비디오 영상으로 AVM 만들고 싶으면, while문 추가하고 그 안에서 이미지 받고 진행하면 된다. 
#
#






# window size (출력되는 화면 크기를 정하는 것으로, AVM 제작과정과는 무관함)
win_sizeX = 416 #500
win_sizeY = 416 #700

# 먼저 TransformMatrix를 구해야한다.
TransformMatrix = mk.getTransformMatrix(win_sizeX, win_sizeY) # win_sizeX, Y는 출력창 크기므로, AVM과 아무상관 없음. 


front_image     = cv2.imread('/home/mkjeong0/a_good/test_img/scene2/undistorted/front.jpg')
left_image      = cv2.imread('/home/mkjeong0/a_good/test_img/scene2/undistorted/left.jpg')
right_image     = cv2.imread('/home/mkjeong0/a_good/test_img/scene2/undistorted/right.jpg')
back_image      = cv2.imread('/home/mkjeong0/a_good/test_img/scene2/undistorted/back.jpg')

width, height, channel = front_image.shape



if front_image is not None and left_image is not None and right_image is not None and back_image is not None:


    ROI_image = mk.getRoiImage2(front_image, left_image, right_image, back_image) #getRoiImage2는 magic number쓴것. 
    AVM_image = mk.AVM(ROI_image, TransformMatrix, win_sizeX, win_sizeY)
    cv2.imshow('AVM_image', AVM_image)
    cv2.waitKey(0)
