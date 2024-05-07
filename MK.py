#-*- coding:utf-8 -*-
import cv2
import numpy as np
import time

##################################################
#  global variables - 사용자가 설정해줘야함.
##################################################
M_DEG2RAD = np.pi / 180.0
M_image_width = 416  #3000 #1760 #1500 #1241 #416
M_image_height = 416  #715 #720 #416 #376
M_fovx = 165 #165 #130 #90

# 차량의 위에서 바라봤을때(차량이 바라보는 방향이 위쪽이 되는 이미지) 그 화면의 모서리들의 월드좌표[m].
# M_는 전역변수를 의미함
M_TL_mp = [13, 13]      # Top left meter point
M_TR_mp = [13, -13]     # Top right meter point
M_TM_mp = [13, 0]
M_BR_mp = [-13, -13]    # bottom right meter point
M_BL_mp = [-13, 13]
M_BM_mp = [-13, 0]
M_RM_mp = [0, -13]
M_LM_mp = [0, 13]
M_car_TL_mp = [2.1315, 0.9875]   # car topleft meter point
M_car_TR_mp = [2.1315, -0.9875]  # car topright meter point
M_car_TM_mp = [2.1315, 0]        # car topmiddle meter point
M_car_BR_mp = [-2.1315, -0.9875]
M_car_BL_mp = [-2.1315, 0.9875]
M_car_BM_mp = [-2.1315, 0]
M_car_LM_mp = [0, 0.9875]
M_car_RM_mp = [0, -0.9875]
##################################################
##################################################


### global variables에 의해 알아서 정해지는곳 ######################
M_f = (M_image_width / 2.0) * (1 / np.tan((M_fovx / 2.0) * M_DEG2RAD))
M_fovy = 2 * np.arctan((M_image_height / 2) / M_f) / M_DEG2RAD  # fovy 는 단순히 fovx * image_height / M_image_width 이거 절대 아니다!! fovy = 2*arctan((image_height/2)/f) 이다.
M_front_TILT = -(90 - M_fovy / 2.0) * M_DEG2RAD # rad and negative value
M_right_TILT = M_front_TILT
M_left_TILT = M_front_TILT
M_back_TILT = M_front_TILT
M_Cx = M_image_width / 2.0
M_Cy = M_image_height / 2.0
##############################################################

# param : [ PAN, TILT, World_CamX, World_CamY, World_CamZ ]
M_front_param = [-np.pi / 2.0, M_front_TILT, 2.1315, 0, 1]
M_right_param = [-np.pi, M_right_TILT, 0, -0.9875, 1]
M_left_param = [0, M_left_TILT, 0, 0.9875, 1]
M_back_param = [np.pi / 2.0, M_back_TILT, -2.1315, 0, 1]


def AVM(ROI_image, TransformMatrix, win_sizeX, win_sizeY): # PAN, TILT [rad]
    # 함수 설명 : ROI 이미지와 변환 Matrix가 주어지면, 와핑(관심영역을 위에서 본것같은 이미지로 펼쳐줌)을 해줌.
    # input   : ROI 이미지, 변환 Matrix, 원하는 출력 윈도우 사이즈.
    # output  : AVM 이미지.
    front_Transformed_image = cv2.warpPerspective(ROI_image[0], TransformMatrix[0], (win_sizeX, win_sizeY))
    left_Transformed_image = cv2.warpPerspective(ROI_image[1], TransformMatrix[1], (win_sizeX, win_sizeY))
    right_Transformed_image = cv2.warpPerspective(ROI_image[2], TransformMatrix[2], (win_sizeX, win_sizeY))
    back_Transformed_image = cv2.warpPerspective(ROI_image[3], TransformMatrix[3], (win_sizeX, win_sizeY))

    goal_image = front_Transformed_image + left_Transformed_image + right_Transformed_image + back_Transformed_image
    return goal_image

def getTransformMatrix(win_sizeX, win_sizeY):
    # 함수 설명 : 전좌우후 카메라의 ROI 이미지를 어떻게 mapping하면 똑바로 잘 펴진 영상이 될 수 있는지, 그 펼치는 과정에 쓰이는 Matrix를 미리 구해놓는 함수.
    #           이 함수를 써서 loop에서 while문 들어가기전에 TransformMatrix 다 구해놓고 loop에서는 매번 이 Matrix를 계속 갖다 쓰기만 하면된다.
    # input  : 원하는 출력 윈도우 사이즈의 가로 세로.
    # output : 전 좌 우 후 TransformMatrix list


    #######################################
    ###############  front  ###############
    #######################################

    # 실제 월드좌표를 입력하여 구해지는 픽셀값들. !!!***** 변환 전 *****!!!
    front_TL_pp = Proj2Img(M_TL_mp[0], M_TL_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    front_TR_pp = Proj2Img(M_TR_mp[0], M_TR_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    front_BR_pp = Proj2Img(M_car_TR_mp[0], M_car_TR_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    front_BL_pp = Proj2Img(M_car_TL_mp[0], M_car_TL_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])

    ##
    ## 2. Transformation matrix 구하기
    ##
    front_src = np.float32(
        [[front_TL_pp[0], front_TL_pp[1]], [front_TR_pp[0], front_TR_pp[1]], [front_BR_pp[0], front_BR_pp[1]],
         [front_BL_pp[0], front_BL_pp[1]]])

    ### **** 여기에 카메라 Translation 이 반영!!
    front_TL_m = abs(M_TL_mp[1] - M_front_param[3])
    front_TR_m = abs(M_front_param[3] - M_TR_mp[1])
    front_BR_m = abs(M_front_param[3] - M_car_BR_mp[1])
    front_BL_m = abs(M_car_BL_mp[1] - M_front_param[3])

    front_T_m = front_TL_m + front_TR_m
    front_H_m = abs(M_TM_mp[0] - M_car_TM_mp[0])

    # 위쪽을 기준으로 잡는다.
    front_T_p = front_T_m / 0.0625  # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로. !!!

    front_TL_p = front_T_p * front_TL_m / front_T_m
    front_BR_p = front_T_p * front_BR_m / front_T_m
    front_BL_p = front_T_p * front_BL_m / front_T_m
    front_H_p = front_T_p * front_H_m / front_T_m

    # 화면 이동 변수들########################################################################################
    GOAL_WINDOW_SIZEX = win_sizeX
    GOAL_WINDOW_SIZEY = win_sizeY
    # MOVE_X = (GOAL_WINDOW_SIZEX - front_T_p) / 2.0  # 248.65289581555314  09.04 20:27 일단 원점으로 바꿈.
    MOVE_X = 0
    MOVE_Y = MOVE_X

    front_dst = np.float32(
        [[MOVE_X, MOVE_Y], [MOVE_X + front_T_p, MOVE_Y], [MOVE_X + front_TL_p + front_BR_p, MOVE_Y + front_H_p],
         [MOVE_X + front_TL_p - front_BL_p, MOVE_Y + front_H_p]])  ##############################

    # 변환.
    front_TransformMatrix = cv2.getPerspectiveTransform(front_src, front_dst)

    #######################################
    ###############  left   ###############
    #######################################
    left_TL_pp = Proj2Img(M_BL_mp[0], M_BL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])
    left_TR_pp = Proj2Img(M_TL_mp[0], M_TL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])
    left_BR_pp = Proj2Img(M_car_TL_mp[0], M_car_TL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])
    left_BL_pp = Proj2Img(M_car_BL_mp[0], M_car_BL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])

    ##
    ## 2. Transformation matrix 구하기
    ## back_param = [np.pi/2.0, back_TILT, -2.1315, 0, 1]
    left_src = np.float32(
        [[(left_TL_pp[0], left_TL_pp[1]), (left_TR_pp[0], left_TR_pp[1]), (left_BR_pp[0], left_BR_pp[1]),
          (left_BL_pp[0], left_BL_pp[1])]])

    # 여기에 카메라 Translation 이 반영!!
    left_TL_m = abs(M_left_param[2] - M_BL_mp[0])
    left_TR_m = abs(M_TL_mp[0] - M_left_param[2])
    left_BR_m = abs(M_car_TL_mp[0] - M_left_param[2])
    left_BL_m = abs(M_left_param[2] - M_car_BL_mp[0])

    left_T_m = left_TL_m + left_TR_m
    left_H_m = abs(M_LM_mp[1] - M_car_LM_mp[1])

    # 위쪽을 기준으로 잡는다.
    left_T_p = left_T_m / 0.0625  # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로. !!!

    left_TL_p = left_T_p * left_TL_m / left_T_m
    left_BR_p = left_T_p * left_BR_m / left_T_m
    left_BL_p = left_T_p * left_BL_m / left_T_m
    left_H_p = left_T_p * left_H_m / left_T_m

    left_dst = np.float32(
        [[MOVE_X, MOVE_Y + left_T_p], [MOVE_X, MOVE_Y],
         [MOVE_X + left_H_p, MOVE_Y + left_T_p - left_TL_p - left_BR_p],
         [MOVE_X + left_H_p, MOVE_Y + left_T_p - left_TL_p + left_BL_p]])

    # 변환.
    left_TransformMatrix = cv2.getPerspectiveTransform(left_src, left_dst)

    #######################################
    ###############  right  ###############
    #######################################
    right_TL_pp = Proj2Img(M_TR_mp[0], M_TR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])
    right_TR_pp = Proj2Img(M_BR_mp[0], M_BR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])
    right_BR_pp = Proj2Img(M_car_BR_mp[0], M_car_BR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])
    right_BL_pp = Proj2Img(M_car_TR_mp[0], M_car_TR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])

    ##
    ## 2. Transformation matrix 구하기
    ## right_param = [-np.pi, right_TILT, 0, -0.9875, 1]
    right_src = np.float32(
        [[(right_TL_pp[0], right_TL_pp[1]), (right_TR_pp[0], right_TR_pp[1]), (right_BR_pp[0], right_BR_pp[1]),
          (right_BL_pp[0], right_BL_pp[1])]])

    # 여기에 카메라 Translation 이 반영!!
    right_TL_m = abs(M_TR_mp[0] - M_right_param[2])
    right_TR_m = abs(M_right_param[2] - M_BR_mp[0])
    right_BR_m = abs(M_right_param[2] - M_car_BR_mp[0])
    right_BL_m = abs(M_car_TL_mp[0] - M_right_param[2])

    right_T_m = right_TL_m + right_TR_m
    right_H_m = abs(M_car_RM_mp[1] - M_RM_mp[1])

    # 위쪽을 기준으로 잡는다.
    right_T_p = right_T_m / 0.0625  # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로. !!!

    right_TL_p = right_T_p * right_TL_m / right_T_m
    right_BR_p = right_T_p * right_BR_m / right_T_m
    right_BL_p = right_T_p * right_BL_m / right_T_m
    right_H_p = right_T_p * right_H_m / right_T_m

    right_dst = np.float32(
        [[MOVE_X + front_T_p, MOVE_Y], [MOVE_X + front_T_p, MOVE_Y + right_T_p],
         [MOVE_X + front_T_p - right_H_p, MOVE_Y + right_TL_p + right_BR_p],
         [MOVE_X + front_T_p - right_H_p, MOVE_Y + right_TL_p - right_BL_p]])

    # 변환.
    right_TransformMatrix = cv2.getPerspectiveTransform(right_src, right_dst)

    #######################################
    ###############  back   ###############
    #######################################
    back_TL_pp = Proj2Img(M_BR_mp[0], M_BR_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])
    back_TR_pp = Proj2Img(M_BL_mp[0], M_BL_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])
    back_BR_pp = Proj2Img(M_car_BL_mp[0], M_car_BL_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])
    back_BL_pp = Proj2Img(M_car_BR_mp[0], M_car_BR_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])

    ##
    ## 2. Transformation matrix 구하기
    ## back_param = [np.pi/2.0, back_TILT, -2.1315, 0, 1]
    back_src = np.float32(
        [[(back_TL_pp[0], back_TL_pp[1]), (back_TR_pp[0], back_TR_pp[1]), (back_BR_pp[0], back_BR_pp[1]),
          (back_BL_pp[0], back_BL_pp[1])]])

    # 여기에 카메라 Translation 이 반영!!
    back_TL_m = abs(M_back_param[3] - M_BR_mp[1])
    back_TR_m = abs(M_BL_mp[1] - M_back_param[3])
    back_BR_m = abs(M_car_BL_mp[1] - M_back_param[3])
    back_BL_m = abs(M_back_param[3] - M_car_BR_mp[1])

    back_T_m = back_TL_m + back_TR_m
    back_H_m = abs(M_car_BM_mp[0] - M_BM_mp[0])

    # 위쪽을 기준으로 잡는다.
    back_T_p = back_T_m / 0.0625  # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로. !!!

    back_TL_p = back_T_p * back_TL_m / back_T_m
    back_BR_p = back_T_p * back_BR_m / back_T_m
    back_BL_p = back_T_p * back_BL_m / back_T_m
    back_H_p = back_T_p * back_H_m / back_T_m

    back_dst = np.float32(
        [[MOVE_X + front_T_p, MOVE_Y + right_T_p], [MOVE_X, MOVE_Y + right_T_p],
         [MOVE_X + front_T_p - back_TL_p - back_BR_p, MOVE_Y + right_T_p - back_H_p],
         [MOVE_X + front_T_p - back_TL_p + back_BL_p, MOVE_Y + right_T_p - back_H_p]])

    # 변환.
    back_TransformMatrix = cv2.getPerspectiveTransform(back_src, back_dst)

    return [front_TransformMatrix, left_TransformMatrix, right_TransformMatrix, back_TransformMatrix]

def getRoiImage(front_image, left_image, right_image, back_image):
    # 함수 설명 : 전좌우후 4개의 화면에서 나오는 영상을 받아, 각각의 영상에서 ROI 쳐야하는 부분만 잘라서(마스킹해서) return하는 함수.
    # input   : 전좌우후 매 프래임 이미지
    # output  : ROI만큼 잘려진 전좌우후 이미지.
    # Proj2Img : 원하는 월드 좌표를 입력하고, 카메라 파라미터들을 입력하면, 이 월드좌표가 카메라 이미지 좌표계에 어느 픽셀에 찍히는지 구해주는 함수.
    #######################################
    ###############  front  ###############
    #######################################
    start = time.time()
    # 실제 월드좌표를 입력하여 구해지는 픽셀값들. !!!***** 변환 전 *****!!!
    front_TL_pp = Proj2Img(M_TL_mp[0], M_TL_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    front_TR_pp = Proj2Img(M_TR_mp[0], M_TR_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    front_BR_pp = Proj2Img(M_car_TR_mp[0], M_car_TR_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    front_BL_pp = Proj2Img(M_car_TL_mp[0], M_car_TL_mp[1], M_front_param[0], M_front_param[1], M_front_param[2], M_front_param[3], M_front_param[4])
    print("Proj2Img_time : ", time.time()-start)

    ##
    ## 1. 다각형 ROI 설정
    ##
    mask = np.zeros(front_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(front_TL_pp[0], front_TL_pp[1]), (front_TR_pp[0], front_TR_pp[1]),
                             (front_BR_pp[0], front_BR_pp[1]), (front_BL_pp[0], front_BL_pp[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied

    channel_count = front_image.shape[2]  # 2번 index가 채널수 = 3
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    front_ROI_image = cv2.bitwise_and(front_image, mask)

    #######################################
    ###############  left   ###############
    #######################################

    left_TL_pp = Proj2Img(M_BL_mp[0], M_BL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])
    left_TR_pp = Proj2Img(M_TL_mp[0], M_TL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])
    left_BR_pp = Proj2Img(M_car_TL_mp[0], M_car_TL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])
    left_BL_pp = Proj2Img(M_car_BL_mp[0], M_car_BL_mp[1], M_left_param[0], M_left_param[1], M_left_param[2], M_left_param[3], M_left_param[4])

    ##
    ## 1. 다각형 ROI 설정
    ## left_param = [0, left_TILT, 0, 0.9875, 1]
    mask = np.zeros(left_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(left_TL_pp[0], left_TL_pp[1]), (left_TR_pp[0], left_TR_pp[1]),
                             (left_BR_pp[0], left_BR_pp[1]), (left_BL_pp[0], left_BL_pp[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    # channel_count = left_image.shape[2]  # 2번 index가 채널수 = 3
    # ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    left_ROI_image = cv2.bitwise_and(left_image, mask)

    #######################################
    ###############  right  ###############
    #######################################

    right_TL_pp = Proj2Img(M_TR_mp[0], M_TR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])
    right_TR_pp = Proj2Img(M_BR_mp[0], M_BR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])
    right_BR_pp = Proj2Img(M_car_BR_mp[0], M_car_BR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])
    right_BL_pp = Proj2Img(M_car_TR_mp[0], M_car_TR_mp[1], M_right_param[0], M_right_param[1], M_right_param[2], M_right_param[3], M_right_param[4])


    ##
    ## 1. 다각형 ROI 설정
    ##
    mask = np.zeros(right_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(right_TL_pp[0], right_TL_pp[1]), (right_TR_pp[0], right_TR_pp[1]),
                             (right_BR_pp[0], right_BR_pp[1]), (right_BL_pp[0], right_BL_pp[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    # channel_count = right_image.shape[2]  # 2번 index가 채널수 = 3
    # ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    right_ROI_image = cv2.bitwise_and(right_image, mask)

    #######################################
    ###############  back   ###############
    #######################################

    back_TL_pp = Proj2Img(M_BR_mp[0], M_BR_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])
    back_TR_pp = Proj2Img(M_BL_mp[0], M_BL_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])
    back_BR_pp = Proj2Img(M_car_BL_mp[0], M_car_BL_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])
    back_BL_pp = Proj2Img(M_car_BR_mp[0], M_car_BR_mp[1], M_back_param[0], M_back_param[1], M_back_param[2], M_back_param[3], M_back_param[4])

    ##
    ## 1. 다각형 ROI 설정
    ##
    mask = np.zeros(back_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(back_TL_pp[0], back_TL_pp[1]), (back_TR_pp[0], back_TR_pp[1]),
                             (back_BR_pp[0], back_BR_pp[1]), (back_BL_pp[0], back_BL_pp[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    # channel_count = back_image.shape[2]  # 2번 index가 채널수 = 3
    # ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    back_ROI_image = cv2.bitwise_and(back_image, mask)

    return [front_ROI_image, left_ROI_image, right_ROI_image, back_ROI_image]

def getRoiImage2(front_image, left_image, right_image, back_image): # getRoiImage의 magic number 버전 함수. 속도 빠름.
    # 함수 설명 : getRoiImage 함수에서 속도만 높인 버젼.
    #           getRoiImage 함수에서 Proj2Img 함수들을 사용하는데, 그 함수안에는 cos, sin 연산, Matrix 곱 연산이 들어있습니다.
    #           그 연산 결과값을 print하여 확인한 후 getRoiImage2 함수에서는 그냥 데이터로 집어넣었습니다.

    #######################################
    ###############  front  ###############
    #######################################
    ##
    ## 1. 다각형 ROI 설정
    ##
    mask = np.zeros(front_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(1243, 270), (1756, 270),
                             (1903, 715), (1096, 714)]], dtype=np.int32) # Proj2Img 함수의 연산 결과를 그냥 데이터로 집어넣은 부분!(함수 처음 주석보세요)
    # fill the ROI so it doesn't get wiped out when the mask is applied

    channel_count = front_image.shape[2]  # 2번 index가 채널수 = 3
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    front_ROI_image = cv2.bitwise_and(front_image, mask)

    #######################################
    ###############  left   ###############
    #######################################
    ##
    ## 1. 다각형 ROI 설정
    ## left_param = [0, left_TILT, 0, 0.9875, 1]
    mask = np.zeros(left_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(1266, 268), (1733, 268),
                             (2370, 715), (629, 714)]], dtype=np.int32) # Proj2Img 함수의 연산 결과를 그냥 데이터로 집어넣은 부분!(함수 처음 주석보세요)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    left_ROI_image = cv2.bitwise_and(left_image, mask)

    #######################################
    ###############  right  ###############
    #######################################
    ##
    ## 1. 다각형 ROI 설정
    ##
    mask = np.zeros(right_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(1266, 268), (1733, 268),
                             (2370, 715), (629, 714)]], dtype=np.int32) # Proj2Img 함수의 연산 결과를 그냥 데이터로 집어넣은 부분!(함수 처음 주석보세요)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    right_ROI_image = cv2.bitwise_and(right_image, mask)

    #######################################
    ###############  back   ###############
    #######################################
    ##
    ## 1. 다각형 ROI 설정
    ##
    mask = np.zeros(back_image.shape, dtype=np.uint8)
    roi_corners = np.array([[(1243, 270), (1756, 270),
                             (1903, 714), (1096, 715)]], dtype=np.int32) # Proj2Img 함수의 연산 결과를 그냥 데이터로 집어넣은 부분!(함수 처음 주석보세요)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    back_ROI_image = cv2.bitwise_and(back_image, mask)

    return [front_ROI_image, left_ROI_image, right_ROI_image, back_ROI_image]


def InvProj2GRD(Ximg, Yimg, PAN, TILT, World_CamX, World_CamY,World_CamZ):  # 2D image coordinates -> 3D World coordinates (0-especially ground points, Zw = 0)
    # input : Ximg, Yimg, PAN(RAD), TILT(RAD), World_CamX, World_CamY, World_CamZ
    # return : np.array[Xw, Yw]
    global M_Cy, M_Cx, M_f
    Zc = (-World_CamZ * M_f) / (-np.cos(TILT) * (Yimg - M_Cy) + M_f * np.sin(TILT))
    Xc = (Ximg - M_Cx) * Zc / M_f
    Yc = (Yimg - M_Cy) * Zc / M_f
    Xw = np.cos(PAN) * Xc - np.sin(PAN) * np.sin(TILT) * Yc - np.sin(PAN) * np.cos(TILT) * Zc + World_CamX
    Yw = np.sin(PAN) * Xc + np.cos(PAN) * np.sin(TILT) * Yc + np.cos(PAN) * np.cos(TILT) * Zc + World_CamY
    # print("Zc : ", Zc, ", Yc : ", Yc, ", Xc : ", Xc , ", Xw : ", Xw, ", Yw : ", Yw) # for test
    result = np.array([Xw, Yw])
    return result


def Proj2Img(Xw, Yw, PAN, TILT, World_CamX, World_CamY, World_CamZ):
    # input : PAN(RAD), TILT(RAD), World_CamX, World_CamY, World_CamZ는 InvProj2GRD와 동일하다.
    # 결과 : print([Ximg, Yimg])
    Zw = 0
    World_Coordinates = np.array([[Xw],[Yw],[Zw]])
    World_Cam_Coordinates = np.array([[World_CamX],[World_CamY],[World_CamZ]])
    RotationMatrix = np.array([[np.cos(PAN), -np.sin(PAN)*np.sin(TILT), -np.sin(PAN)*np.cos(TILT)],
                               [np.sin(PAN), np.cos(PAN)*np.sin(TILT), np.cos(PAN)*np.cos(TILT)],
                               [0, -np.cos(TILT), np.sin(TILT)]])
    Cam_Coordinates = np.dot(np.linalg.inv(RotationMatrix),(World_Coordinates - World_Cam_Coordinates)) # Xc, Yc, Zc
    Ximg = Cam_Coordinates[0,0] / Cam_Coordinates[2,0] * M_f + M_Cx
    Yimg = Cam_Coordinates[1,0] / Cam_Coordinates[2,0] * M_f + M_Cy
    return [Ximg, Yimg]

def Proj2Img_CxCyf(Xw, Yw, PAN, TILT, World_CamX, World_CamY, World_CamZ, Cx, Cy, f):
    # input : PAN(RAD), TILT(RAD), World_CamX, World_CamY, World_CamZ는 InvProj2GRD와 동일하다.
    # 결과 : print([Ximg, Yimg])
    Zw = 0
    World_Coordinates = np.array([[Xw],[Yw],[Zw]])
    World_Cam_Coordinates = np.array([[World_CamX],[World_CamY],[World_CamZ]])
    RotationMatrix = np.array([[np.cos(PAN), -np.sin(PAN)*np.sin(TILT), -np.sin(PAN)*np.cos(TILT)],
                               [np.sin(PAN), np.cos(PAN)*np.sin(TILT), np.cos(PAN)*np.cos(TILT)],
                               [0, -np.cos(TILT), np.sin(TILT)]])
    Cam_Coordinates = np.dot(np.linalg.inv(RotationMatrix),(World_Coordinates - World_Cam_Coordinates)) # Xc, Yc, Zc
    Ximg = Cam_Coordinates[0,0] / Cam_Coordinates[2,0] * f + Cx
    Yimg = Cam_Coordinates[1,0] / Cam_Coordinates[2,0] * f + Cy
    return [Ximg, Yimg]

'''
# def getCrossPoints(lines, points):
#     # 함수 설명 : combination 개념으로 2개의 직선을 고르면 교점을 구하는 함수
#     # input : lines(HoughLines 리턴값 : array.... not list...), points(교점을 담을 배열)
#     # output : points(교점을 담은 배열)

#     def generate(chosen):
#         if len(chosen) == 2:
#             r1, theta1 = (chosen[0])[0]
#             r2, theta2 = (chosen[1])[0]
#             x = ( -r1 * (np.cos(theta1) * np.tan(theta2) + np.sin(theta1) * np.tan(theta1) * np.tan(theta2)) \
#                 + r2 * (np.cos(theta2) * np.tan(theta1) + np.sin(theta2) * np.tan(theta1) * np.tan(theta2)) ) / (np.tan(theta1) - np.tan(theta2))
#             y = - (x - r1 * np.cos(theta1))/np.tan(theta1) + r1 * np.sin(theta1)
#             points.append([x,y])

#     	# shuffling
#         start = ((np.where(lines == chosen[-1]))[0])[0] + 1 if chosen else 0
#         for nxt in range(start, len(lines)):
#             chosen.append(lines[nxt])
#             generate(chosen)
#             chosen.pop()
#     generate([])

#     return points


# def BFS(edges, queue):
#     deltas = ((-1,0),(1,0),(0,-1),(0,1))
#     while queue:
#         x, y = queue.popleft()
#         for dx, dy in deltas:
#             nx, ny = x +dx, y + dy
#             if 0<=nx<edges.shape[1] and 0<=ny<edges.shape[0]:
#                 if edges[ny][nx] == 0 and edges[y][x]>50: #방문하지 않는 노드라면.
#                     edges[ny][nx] = edges[y][x] * (0.4)
#                     queue.append((nx,ny))
#     return edges



def ROI2TransformMatrix(queue_item, topleft, topright, bottomright, bottomleft, PAN, TILT, World_CamX, World_CamY, World_CamZ): # PAN, TILT [rad]
    global M_image_width, M_image_height
    # 함수 설명 : 그냥 ROI 대칭 네모박스만 되는 ver.
    # 이 4개의 인자는, 전체 이미지 평면에서의 좌표쌍[x,y]을 의미한다.
    # topleft       = [topleft_x,       topleft_y]
    # topright      = [topright_x,      topright_y]
    # bottomright   = [bottomright_x,   bottomright_y]
    # bottomleft    = [bottomleft_x,    bottomleft_y]
    # PAN, TILT : rad값을 넣어야함!!
    # World_CamX, World_CamY, World_CamZ : 월드좌표계 상에서 카메라의 위치.

    # 1. ROI 설정.
    ROI = queue_item[int(round(topleft[1])):int(round(bottomleft[1]+1)), int(round(topleft[0])):int(round(topright[0]+1))]  # 반대로 height, width 순서. 208: image_height/2.0

    src = np.float32([[0, 0], [abs(topright[0]-topleft[0]), 0], [abs(topright[0]-topleft[0]), abs(bottomright[1]-topright[1])], [0, abs(bottomright[1]-topright[1])]])

    # 실제 좌표 구하기.
    topleft_world       = InvProj2GRD(topleft[0], topleft[1],           PAN, TILT, World_CamX, World_CamY, World_CamZ)
    topright_world      = InvProj2GRD(topright[0],topright[1],          PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomleft_world    = InvProj2GRD(bottomleft[0],bottomleft[1],      PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomright_world   = InvProj2GRD(bottomright[0],bottomright[1],    PAN, TILT, World_CamX, World_CamY, World_CamZ)


    bottom_meter = abs(bottomleft_world[1] - bottomright_world[1])
    top_meter = abs(topleft_world[1] - topright_world[1])
    height_meter = abs(topleft_world[0] - bottomleft_world[0])

    print("topmeter : " , top_meter, "heightmeter : ", height_meter)

    # 1. 또는 2. 를 선택하세요. 나머지는 주석처리
    # 1. 위쪽을 기준으로 잡을때
    top_pixel = 416                     # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로. !!! wow~~~
    bottom_pixel = top_pixel * bottom_meter / top_meter
    height_pixel = top_pixel * height_meter / top_meter
    dst = np.float32([[0, 0], [top_pixel, 0], [top_pixel / 2.0 + bottom_pixel / 2, height_pixel],[top_pixel / 2.0 - bottom_pixel / 2, height_pixel]])  ##############################
    print("bottom_pixel : ", bottom_pixel, " height_pixel : ", height_pixel)

    # 2. 아랫쪽을 기준으로 잡을때
    # bottom_pixel = abs(bottomright[0]-bottomleft[0])
    # top_pixel = bottom_pixel * top_meter / bottom_meter
    # height_pixel = bottom_pixel * height_meter / bottom_meter
    # dst = np.float32([[0, 0], [top_pixel, 0], [top_pixel / 2.0 + bottom_pixel / 2, height_pixel],[top_pixel / 2.0 - bottom_pixel / 2, height_pixel]])  ##############################

    # 변환.
    TransformMatrix = cv2.getPerspectiveTransform(src, dst)

    # ROI 한번 확인해보기
    cv2.imshow("ROI", ROI)
    print("가로 1pixel 당 거리 : ", top_meter / top_pixel, "세로 1pixel 당 거리 : ", height_meter / height_pixel)

    return cv2.warpPerspective(ROI, TransformMatrix,(int(round(top_pixel)), int(round(height_pixel))))
def ROI2TransformMatrix2(queue_item, topleft, topright, bottomright, bottomleft, PAN, TILT, World_CamX, World_CamY, World_CamZ): # PAN, TILT [rad]
    global M_image_width, M_image_height
    # 함수 설명 : 다각형 ROI 되는 ver.
    # 이 4개의 인자는, 전체 이미지 평면에서의 좌표쌍[x,y]을 의미한다.
    # topleft       = [topleft_x,       topleft_y]
    # topright      = [topright_x,      topright_y]
    # bottomright   = [bottomright_x,   bottomright_y]
    # bottomleft    = [bottomleft_x,    bottomleft_y]
    # PAN, TILT : rad값을 넣어야함!!
    # World_CamX, World_CamY, World_CamZ : 월드좌표계 상에서 카메라의 위치.

    ## 1. 다각형 ROI 설정. ##
    mask = np.zeros(queue_item.shape, dtype=np.uint8)
    roi_corners = np.array([[(topleft[0],topleft[1]), (topright[0],topright[1]), (bottomright[0],bottomright[1]),(bottomleft[0],bottomleft[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = queue_item.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    masked_image = cv2.bitwise_and(queue_item, mask)


    ## 2. Transformation matrix 구하기. ##
    src = np.float32([[topleft[0],topleft[1]], [topright[0],topright[1]], [bottomright[0],bottomright[1]], [bottomleft[0],bottomleft[1]]])

    # 실제 좌표 구하기.
    topleft_world       = InvProj2GRD(topleft[0], topleft[1],           PAN, TILT, World_CamX, World_CamY, World_CamZ)
    topright_world      = InvProj2GRD(topright[0],topright[1],          PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomleft_world    = InvProj2GRD(bottomleft[0],bottomleft[1],      PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomright_world   = InvProj2GRD(bottomright[0],bottomright[1],    PAN, TILT, World_CamX, World_CamY, World_CamZ)


    bottom_meter = abs(bottomleft_world[1] - bottomright_world[1])
    top_meter = abs(topleft_world[1] - topright_world[1])
    height_meter = abs(topleft_world[0] - bottomleft_world[0])

    print("topmeter : " , top_meter, "heightmeter : ", height_meter)

    # 1. 또는 2. 를 선택하세요. 나머지는 주석처리
    # 1. 위쪽을 기준으로 잡을때
    top_pixel = 416                     # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로. !!! wow~~~
    bottom_pixel = top_pixel * bottom_meter / top_meter
    height_pixel = top_pixel * height_meter / top_meter
    dst = np.float32([[0, 0], [top_pixel, 0], [top_pixel / 2.0 + bottom_pixel / 2, height_pixel],[top_pixel / 2.0 - bottom_pixel / 2, height_pixel]])  ##############################
    print("bottom_pixel : ", bottom_pixel, " height_pixel : ", height_pixel)


    # 2. 아랫쪽을 기준으로 잡을때
    # bottom_pixel = abs(bottomright[0]-bottomleft[0])
    # top_pixel = bottom_pixel * top_meter / bottom_meter
    # height_pixel = bottom_pixel * height_meter / bottom_meter
    # dst = np.float32([[0, 0], [top_pixel, 0], [top_pixel / 2.0 + bottom_pixel / 2, height_pixel],[top_pixel / 2.0 - bottom_pixel / 2, height_pixel]])  ##############################

    # 변환.
    TransformMatrix = cv2.getPerspectiveTransform(src, dst)

    # ROI 한번 확인해보기
    cv2.imshow("masked_image", masked_image)
    print("가로 1pixel 당 거리 : ", top_meter / top_pixel, "세로 1pixel 당 거리 : ", height_meter / height_pixel)

    return cv2.warpPerspective(masked_image, TransformMatrix,(int(round(top_pixel)), int(round(height_pixel))))
'''


def ROI2TransformMatrix3(queue_item, topleft, topright, bottomright, bottomleft, PAN, TILT, World_CamX, World_CamY, World_CamZ): # PAN, TILT [rad]
    global M_image_width, M_image_height
    # 함수 설명 : 다각형 뿐만 아니라, Yimg의 높이만 같다면, 중심이 가운데 있지 않아도 되는 만능 ver.
    # 이 4개의 인자는, 전체 이미지 평면에서의 좌표쌍[x,y]을 의미한다.
    # topleft       = [topleft_x,       topleft_y]
    # topright      = [topright_x,      topright_y]
    # bottomright   = [bottomright_x,   bottomright_y]
    # bottomleft    = [bottomleft_x,    bottomleft_y]
    # PAN, TILT : rad값을 넣어야함!!
    # World_CamX, World_CamY, World_CamZ : 월드좌표계 상에서 카메라의 위치.

    ## 1. 다각형 ROI 설정. ##
    mask = np.zeros(queue_item.shape, dtype=np.uint8)
    roi_corners = np.array([[(topleft[0],topleft[1]), (topright[0],topright[1]), (bottomright[0],bottomright[1]),(bottomleft[0],bottomleft[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = queue_item.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    masked_image = cv2.bitwise_and(queue_item, mask)
    # print("shape : ", masked_image.shape) # 이걸 보면 알수있듯, 픽셀값에 실수 집어넣어도 정수로 변환돼서 지 알아서 한다.

    ## 2. Transformation matrix 구하기. ##
    src = np.float32([[topleft[0],topleft[1]], [topright[0],topright[1]], [bottomright[0],bottomright[1]], [bottomleft[0],bottomleft[1]]])

    # 실제 좌표 구하기.
    topleft_world       = InvProj2GRD(topleft[0], topleft[1],           PAN, TILT, World_CamX, World_CamY, World_CamZ)
    topright_world      = InvProj2GRD(topright[0],topright[1],          PAN, TILT, World_CamX, World_CamY, World_CamZ)
    topmiddle_world     = InvProj2GRD(M_image_width/2.0, 0,               PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomleft_world    = InvProj2GRD(bottomleft[0],bottomleft[1],      PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomright_world   = InvProj2GRD(bottomright[0],bottomright[1],    PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottommiddle_world  = InvProj2GRD(M_image_width / 2.0, M_image_height, PAN, TILT, World_CamX, World_CamY, World_CamZ)

    topleft_meter = abs(topleft_world[1]-topmiddle_world[1])
    topright_meter = abs(topmiddle_world[1]-topright_world[1])
    bottomright_meter = abs(bottommiddle_world[1]-bottomright_world[1])
    bottomleft_meter = abs(bottomleft_world[1]-bottommiddle_world[1])

    bottom_meter = bottomleft_meter + bottomright_meter
    top_meter = topleft_meter + topright_meter
    height_meter = abs(topleft_world[0] - bottomleft_world[0])


    # 위쪽을 기준으로 잡는다.
    top_pixel = top_meter/0.0625  # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로.
    print("top_pixel : ", top_pixel)

    topleft_pixel = top_pixel * topleft_meter / top_meter
    topright_pixel = top_pixel * topright_meter / top_meter
    bottomright_pixel = top_pixel * bottomright_meter / top_meter
    bottomleft_pixel = top_pixel * bottomleft_meter / top_meter
    height_pixel = top_pixel * height_meter / top_meter
    dst = np.float32([[0, 0], [top_pixel, 0], [topleft_pixel + bottomright_pixel, height_pixel],[topleft_pixel - bottomleft_pixel, height_pixel]])  ##############################

    # 변환.
    TransformMatrix = cv2.getPerspectiveTransform(src, dst)

    # ROI 한번 확인해보기
    cv2.imshow("masked_image", masked_image)
    print("가로 1pixel 당 거리 : ", top_meter / top_pixel, "세로 1pixel 당 거리 : ", height_meter / height_pixel)

    return cv2.warpPerspective(masked_image, TransformMatrix,(int(round(top_pixel)), int(round(height_pixel))))

def ROI2TransformMatrix3_CxCyf(queue_item, image_width, image_height, topleft, topright, bottomright, bottomleft, PAN, TILT, World_CamX, World_CamY, World_CamZ, Cx, Cy, f): # PAN, TILT [rad]
    # 함수 설명 : 다각형 뿐만 아니라, Yimg의 높이만 같다면, 중심이 가운데 있지 않아도 되는 만능 ver.
    # 이 4개의 인자는, 전체 이미지 평면에서의 좌표쌍[x,y]을 의미한다.
    # topleft       = [topleft_x,       topleft_y]
    # topright      = [topright_x,      topright_y]
    # bottomright   = [bottomright_x,   bottomright_y]
    # bottomleft    = [bottomleft_x,    bottomleft_y]
    # PAN, TILT : rad값을 넣어야함!!
    # World_CamX, World_CamY, World_CamZ : 월드좌표계 상에서 카메라의 위치.

    ## 1. 다각형 ROI 설정. ##
    mask = np.zeros(queue_item.shape, dtype=np.uint8)
    roi_corners = np.array([[(topleft[0],topleft[1]), (topright[0],topright[1]), (bottomright[0],bottomright[1]),(bottomleft[0],bottomleft[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = queue_item.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # apply the mask
    masked_image = cv2.bitwise_and(queue_item, mask)
    # print("shape : ", masked_image.shape) # 이걸 보면 알수있듯, 픽셀값에 실수 집어넣어도 정수로 변환돼서 지 알아서 한다.

    ## 2. Transformation matrix 구하기. ##
    src = np.float32([[topleft[0],topleft[1]], [topright[0],topright[1]], [bottomright[0],bottomright[1]], [bottomleft[0],bottomleft[1]]])

    # 실제 좌표 구하기.
    topleft_world       = InvProj2GRD(topleft[0], topleft[1],           PAN, TILT, World_CamX, World_CamY, World_CamZ)
    topright_world      = InvProj2GRD(topright[0],topright[1],          PAN, TILT, World_CamX, World_CamY, World_CamZ)
    topmiddle_world     = InvProj2GRD(image_width/2.0, 0,               PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomleft_world    = InvProj2GRD(bottomleft[0],bottomleft[1],      PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottomright_world   = InvProj2GRD(bottomright[0],bottomright[1],    PAN, TILT, World_CamX, World_CamY, World_CamZ)
    bottommiddle_world  = InvProj2GRD(image_width / 2.0, image_height, PAN, TILT, World_CamX, World_CamY, World_CamZ)

    topleft_meter = abs(topleft_world[1]-topmiddle_world[1])
    topright_meter = abs(topmiddle_world[1]-topright_world[1])
    bottomright_meter = abs(bottommiddle_world[1]-bottomright_world[1])
    bottomleft_meter = abs(bottomleft_world[1]-bottommiddle_world[1])

    bottom_meter = bottomleft_meter + bottomright_meter
    top_meter = topleft_meter + topright_meter
    height_meter = abs(topleft_world[0] - bottomleft_world[0])


    # 위쪽을 기준으로 잡는다.
    top_pixel = top_meter/0.0625  # 이거를 기준으로 화면의 크기를 조정할 수 있다. 내 마음대로.
    print("top_pixel : ", top_pixel)

    topleft_pixel = top_pixel * topleft_meter / top_meter
    topright_pixel = top_pixel * topright_meter / top_meter
    bottomright_pixel = top_pixel * bottomright_meter / top_meter
    bottomleft_pixel = top_pixel * bottomleft_meter / top_meter
    height_pixel = top_pixel * height_meter / top_meter
    dst = np.float32([[0, 0], [top_pixel, 0], [topleft_pixel + bottomright_pixel, height_pixel],[topleft_pixel - bottomleft_pixel, height_pixel]])  ##############################

    # 변환.
    TransformMatrix = cv2.getPerspectiveTransform(src, dst)

    # ROI 한번 확인해보기
    cv2.imshow("masked_image", masked_image)
    print("가로 1pixel 당 거리 : ", top_meter / top_pixel, "세로 1pixel 당 거리 : ", height_meter / height_pixel)

    return cv2.warpPerspective(masked_image, TransformMatrix,(int(round(top_pixel)), int(round(height_pixel))))
