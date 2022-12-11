
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from retinaface import RetinaFace
from PIL import Image, ImageEnhance

base_root = "data"

def authenticate_image(area_threshold):
    """
    check if the cropped image has some features. in the case of are, and image ratio
    """
    is_authentic = True
    if area_threshold < .6:
        is_authentic = False            
    return is_authentic

def get_frame_info(img):
    """
    try to find the largest  contour after image processing 
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 220, 240, cv2.THRESH_BINARY_INV )

    cnts = cv2.findContours(np.uint8(threshed), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    largestCnt = []
    for cnt in cnts:
        if (len(cnt) > len(largestCnt)):
            largestCnt = cnt

    x,y,w,h = cv2.boundingRect(largestCnt)
    rect = cv2.minAreaRect(largestCnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    cropped_image = img[y:y+h, x:x+w]

    return rect, cropped_image, box, w, h


def facial_cordination(image):
    resp = RetinaFace.detect_faces(image)
    angle = 0
    try:
        left_eye = resp["face_1"]["landmarks"]['left_eye']
        right_eye = resp["face_1"]["landmarks"]['right_eye']
        nose = resp['face_1']['landmarks']['nose']
        
        if nose[1] > left_eye[1] and nose[1] > right_eye[1]:
            angle = angle
        elif nose[1] < left_eye[1] and nose[1] > right_eye[1]:
            angle = angle + 90
        elif nose[1] > left_eye[1] and nose[1] < right_eye[1]:
            angle = angle + 270
        elif nose[1] < left_eye[1] and nose[1] < right_eye[1]:
            angle = angle + 180
    except:
        print(' cant detect face ')
    
    return angle


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] 
    image_center = (width/2, height/2) 

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),  borderValue=(255,255,255))

    return rotated_mat

def scaler(image):
    scled_image = cv2.resize(image , (850,550))
    return scled_image

def get_ID(image):
    image_ID = image[140:190 , 450:690 , :]
    image_ID = cv2.resize(image_ID , (1000,250))

    return image_ID

def detect_frame(img_path):
    file_name = img_path.split("\\")[-1]
    img = cv2.imread(img_path)
    rect, cropped_image, box, width, height = get_frame_info(img)
    angle = rect[2]
    
    upper_edge = np.sqrt((box[0][0] - box[1][0]) ** 2  + (box[0][1] - box[1][1]) ** 2)
    side_edge = np.sqrt((box[1][0] - box[2][0]) ** 2  + (box[1][1] - box[2][1]) ** 2)

    if upper_edge > side_edge:
        angle = angle - 90
    else:
        angle = angle
    
    org_w, org_h = img.shape[:2]
    real_area = org_w * org_h
    crop_area = height * width

    permit = authenticate_image(area_threshold = crop_area/real_area )
    
    if not permit : 
        cropped_image = img
        angle = 0

    ## ---------------------------------------------------------------------------
    rotated_image = rotate_image(mat=cropped_image , angle=angle)

    ## ---------------------------------------------------------------------------
    angle_2 = facial_cordination(rotated_image)
    final_rotation = rotate_image(mat=rotated_image , angle=angle_2)
    
    height, width = final_rotation.shape[:2]
    image_center = (width/2, height/2)
    
    width = max(upper_edge, side_edge)
    height = min(upper_edge, side_edge)
    start_x, end_x, start_y, end_y = int(image_center[0] - width/2) , int(image_center[0] + width/2) , int(image_center[1] - height/2) , int(image_center[1] + height/2)

    frame_image = final_rotation[start_y : end_y , start_x : end_x , : ]
    if not permit : 
        frame_image = final_rotation

    cv2.imwrite(os.path.join(base_root , f"frame\\{file_name}"), frame_image)

    scaled_image = scaler(frame_image)
    ID_image = get_ID(scaled_image)

    cv2.imwrite(os.path.join(base_root , f"ID\\{file_name}"), ID_image)

    print(' ----------> frame detected without any problems <----------')

    return frame_image