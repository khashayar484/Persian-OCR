from keras.models import load_model
import os
import cv2
import numpy as np
import pandas as pd

model = load_model("model\\ID_prediction_model.h5")

def prediction_pipeline(directory):
    prediction_info = pd.DataFrame()
    img_path = os.path.join(directory)
    for file in os.listdir(img_path):
        image_path =  os.path.join(img_path , file)
        img = cv2.imread(image_path)
        face_img = cv2.resize(img, (32, 32))
        face_img = face_img.reshape(1,32,32,3)

        prediction_result = model.predict(face_img)
        classes_x = np.argmax(prediction_result,axis=1)
        if int(classes_x) == 10:
            print('---------> delte it')
        prediction_info = prediction_info.append({"prediction" : int(classes_x[0]) , 'distance' : int(file.replace(".jpg" , ""))} , ignore_index = True)
    
    prediction_info = prediction_info.sort_values(by = ['distance'] , ascending=True)
    national_id_prediction = "".join([str(int(x)) for x in prediction_info['prediction']])
    
    print('-------> final result is ' , national_id_prediction)
    
    return national_id_prediction

if __name__ == "__main__":
    base_root = 'data'
    number_dir = "Number"
    file_name = "04.jpg"
    numbers_directory = os.path.join(base_root, number_dir, file_name)
    print(os.path.join(base_root, number_dir, file_name))
    national_id_prediction = prediction_pipeline(numbers_directory)

