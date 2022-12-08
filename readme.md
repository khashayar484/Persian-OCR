# Persian OCR.
this project aim to get national ID code from its images. the main problem is that the national ID can be located anywhere on a paper with irrgular size and scale, so here, first try to locate frame of the card, align it after than cropping the nummber, scale it and then get the ID numbers, after that predict each number by CNN model.

## Table of contents

the main processes of predicting ID card mention below.

### Frame detection: rotation, get alpha, find
because images can be located anywhere, first we need to align the frame then crop the ID number based on its location. to do this 
after inverting the image BGR2GRAY with OpenCv then grab the largest contour which is ID card's frame, align the frame which gets from 
the conjuction between edge and the center, this method is more precisly than aligning ID card by a it's face.

![OCR_me](https://user-images.githubusercontent.com/54494078/206193822-95c7348e-ef3f-405e-8295-ae8050f3e367.jpg)

above image shows one of the sample of ID card locate on paper. as you can see a right image shows the input immage and the left one shows the trimed image which show the frame after scaling and cropping.  

![OCR_1](https://user-images.githubusercontent.com/54494078/206299845-3e1d01cf-a29a-4c13-a458-ef120de4c1f7.jpg)
![OCR_2](https://user-images.githubusercontent.com/54494078/206300400-05062a05-e197-4118-b171-b9959e0ed39f.jpg)

some images like above cropped before, so here we need to set autentications for cropping images or not and align the result.

### ID detection 
after grabbing frame and detecting the ID number, we need to get numbers from ID. inorder to do this the OTSU threshold is used to grab contour 
from image some flags is implemented to control if the contour is autenticated, these numbers is used as inputs of CNN model.

### model prediction
altought some models like AdaBoost, KNN, Bayes with its dimensional reduction, PCA,LDA can get the good result in prediction here CNN models is used and get the 100% 
accuracy with just afew ID card sample. here we use CNN model with cross entropy loss because we dealing with classification problem.
<image accuracy per each epoch >

![Capture](https://user-images.githubusercontent.com/54494078/206301499-babc4d6e-2272-46fb-a5d8-bee076c71a5f.jpg)

above picture shows the accuracy and loss for each eopches.

### libraries
the entire processes are shown in html with javascript, ajax is used to transfer base64 image from server which is flask to client and viceversa.
OpenCV and PIL packages is used for image preprocessing with retinaface package for detecting face for better alignment.
at the end keras is used for implementing the CNN model.

