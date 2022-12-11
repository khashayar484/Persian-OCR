
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import shutil

base_root = "data"


def check_existance(directory):
    """
    remove folder if exists else create it
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        os.makedirs(directory)

def get_numbers(original_image, contour ,number_directory):
    try:
        x,y,w,h = cv2.boundingRect(contour)
        
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        
        ROI = original_image[y:y+h, x:x+w]

        area = w*h
        distance =+ cX
        
        if area > 1500 and area < 8500:
            ROI = cv2.resize(ROI , (100,250))
            im = Image.fromarray(ROI)
            im.save(os.path.join(number_directory , f"{str(int(distance))}.jpg"), quality=100, subsampling=0)

        elif area> 8500 and area < 17000:
            width = w
            half = width // 2
            
            left_one = original_image[y:y+h, x:x+half]
            right_one = original_image[y:y+h, x+half:x+w]
            
            left_one = cv2.resize(left_one , (100,250))
            left_one = Image.fromarray(left_one)
            left_one.save(os.path.join(number_directory , f"{str(int(distance) - 5)}.jpg"), quality=100, subsampling=0)

            right_one = cv2.resize(right_one , (100,250))
            right_one = Image.fromarray(right_one)
            right_one.save(os.path.join(number_directory , f"{str(int(distance) + 5)}.jpg"), quality=100, subsampling=0)

        elif area > 17000 and area < 25000:
            width = w
            third = width // 3
            
            left_one = original_image[y:y+h, x:x+third]
            mid_one = original_image[y:y+h, x+third:x+2*third]
            right_one = original_image[y:y+h, x+2*third:x+w]
            
            left_one = cv2.resize(left_one , (100,250))
            left_one = Image.fromarray(left_one)
            left_one.save(os.path.join(number_directory , f"{str(int(distance) - 5)}.jpg"), quality=100, subsampling=0)

            mid_one = cv2.resize(mid_one , (100,250))
            mid_one = Image.fromarray(mid_one)
            mid_one.save(os.path.join(number_directory , f"{str(int(distance) )}.jpg"), quality=100, subsampling=0)

            right_one = cv2.resize(right_one , (100,250))
            right_one = Image.fromarray(right_one)
            right_one.save(os.path.join(number_directory , f"{str(int(distance) + 5)}.jpg"), quality=100, subsampling=0)

        elif area > 26000:
            width = w
            quarter = width // 4 
            
            left_one = original_image[y:y+h, x:x+quarter]
            mid_one = original_image[y:y+h, x+quarter:x+2*quarter]
            right_one = original_image[y:y+h, x+2*quarter:x+3*quarter]
            after_left_one = original_image[y:y+h, x+3*quarter:x+w]

            after_left_one = cv2.resize(after_left_one , (100,250))
            after_left_one = Image.fromarray(after_left_one)
            after_left_one.save(os.path.join(number_directory , f"{str(int(distance) -10 )}.jpg"), quality=100, subsampling=0)

            left_one = cv2.resize(left_one , (100,250))
            left_one = Image.fromarray(left_one)
            left_one.save(os.path.join(number_directory , f"{str(int(distance) - 5)}.jpg"), quality=100, subsampling=0)

            mid_one = cv2.resize(mid_one , (100,250))
            mid_one = Image.fromarray(mid_one)
            mid_one.save(os.path.join(number_directory , f"{str(int(distance))}.jpg"), quality=100, subsampling=0)

            right_one = cv2.resize(right_one , (100,250))
            right_one = Image.fromarray(right_one)
            right_one.save(os.path.join(number_directory , f"{str(int(distance) + 5)}.jpg"), quality=100, subsampling=0)
    except:
        print(" couldn't detect number ")


def numebr_detection(ID_path):
    file_name = ID_path.split("\\")[-1]
    ID_path = os.path.join(base_root , f"ID\\{file_name}")
    folder_name = file_name
    Number_directory = os.path.join(base_root , "Number" , folder_name)

    check_existance(Number_directory)

    image = cv2.imread(ID_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        get_numbers(original_image=image.copy(), contour = c, number_directory = Number_directory)
    
