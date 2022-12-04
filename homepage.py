

import os
import io
import base64
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

from preprocessing import frame_detection, number_detection
from model import image_predictor


app = Flask(__name__,  template_folder = "UI")


@app.route("/", methods = ['GET' , 'POST'])
def homepage():
    return render_template("homepage.html")

def get_encoded_image(img_path):
    img = Image.open(img_path, mode = 'r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr , format = 'PNG')
    my_encoded_image = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')

    return my_encoded_image

@app.route("/image" , methods = ['GET' , 'POST'])
def get_image():
    base_root = 'data'
    if request.method == 'POST':
        file = request.files['pic']
        file_name = secure_filename(file.filename)
        upload_dir = 'uploaded_image'
        
        ## -------------> detect frame
        img_path = os.path.join(os.path.join(base_root, upload_dir, file_name))
        file.save(img_path)
        frame_detection.detect_frame(img_path = img_path)
        img = get_encoded_image(os.path.join(base_root , 'frame' , file_name))

        ## -------------> detect ID 
        ID_path = "ID"
        print(base_root, ID_path, file_name)
        number_detection.numebr_detection(os.path.join(base_root, ID_path, file_name))
        
        ## -------------> predict 
        number_dir = "Number"
        national_id_prediction = image_predictor.prediction_pipeline(os.path.join(base_root, number_dir, file_name))
        
        return jsonify({'national_id' : national_id_prediction , 'output_image' : img})


if __name__ == "__main__":
    app.run(debug = False)