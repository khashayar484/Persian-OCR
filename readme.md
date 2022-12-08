# Persian OCR.
this project aim to get national ID code from its images. the main problem is that the national ID can be located anywhere on a paper with irrgular size and scale, so here, first try to locate frame of the card, align it after than cropping the nummber, scale it and then get the ID numbers, after that predict each number by CNN model.



Titles and internal titles : Persian National ID detection 

![OCR_1](https://user-images.githubusercontent.com/54494078/206299845-3e1d01cf-a29a-4c13-a458-ef120de4c1f7.jpg)
![OCR_2 (1)](https://user-images.githubusercontent.com/54494078/206300400-05062a05-e197-4118-b171-b9959e0ed39f.jpg)


## Table of contents

the main processes of predicting ID card mention below.

### Frame detection: rotation, get alpha, find
because images can be located anywhere, first we need to align the frame then crop the ID number based on its location. to do this 
after inverting the image BGR2GRAY with OpenCv then grab the largest contour which is ID card's frame, align the frame which gets from 
the conjuction between edge and the center, this method is more precisly than aligning ID card by a it's face.

![OCR_me](https://user-images.githubusercontent.com/54494078/206193822-95c7348e-ef3f-405e-8295-ae8050f3e367.jpg)

above image shows one of the sample of ID card locate on paper. the 

. ID detection 
after grabbing frame and detecting the ID number first scale image and then grab each numbers from ID. inorder to do this the OTSU threshold is used to grab contour 
from image some flags is implemented to control if the contour is autenticated or not after that, these numbers is used as inputs of CNN model.
. model prediction
altought some models like AdaBoost, KNN, Bayes with its dimensional reduction, PCA,LDA can get the good result in prediction here CNN models is used and get the 100% 
accuracy with just afew ID card sample. for better result use another category "10" when model can't predict the number or when the incorrect inputs gets to the network
<image accuracy per each epoch >
<img>  </img>
the model structure.
![Capture](https://user-images.githubusercontent.com/54494078/206301499-babc4d6e-2272-46fb-a5d8-bee076c71a5f.jpg)

. Visualziation: 
the entire processes are shown in html with javascript, ajax is used to transfer base64 image from server which is flask to client and viceversa.

Prerequisites
	Python 3+
	Opencv 3.4.+
	Numpy
	Scikit-Image
	Tensorflow
	keras
	Ajax

Other information
