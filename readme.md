# Persian OCR.

This project aims to get a national ID code from its images. the main problem is that the national ID can be located anywhere on a paper with irregular size and scale, so here, first try to locate the frame of the card, align it after cropping the number, scale it and then get the ID numbers, after that predict each number by CNN model.

## Table of contents:

The main processes of predicting ID card is mentioned below.

### Frame detection and alignment:

Because images can be located anywhere, first we need to align the frame and then crop the ID number based on its location. to do this 
after inverting the image BGR2GRAY with OpenCV, grab the largest contour which is the ID card's frame, and align the frame which gets from 
the conjunction between the edge and the center, this method is more precise than aligning an ID card by its face.

<img src="https://user-images.githubusercontent.com/54494078/207532768-998c2398-b4b9-4965-a523-e53b020cbfc8.jpg" width="1100" height="500" align = 'center' >

![OCR_4 (3)](https://user-images.githubusercontent.com/54494078/207544610-c03b3a0a-c652-487d-85d0-a68fa53360aa.jpg)


The above image shows one of the samples of ID cards locate on paper. as you can see the right image shows the input image and the left one shows the trim image which shows the frame after scaling and cropping.  

<img src="https://user-images.githubusercontent.com/54494078/207532858-57130dcc-3545-4270-869a-d44a1c41398d.jpg" width="1100" height="600" align = 'center' >

Some images like the above were cropped before, so here we need to set authentications for cropping images or not, then align the result.

### ID detection:

After grabbing the frame and detecting the ID number, we need to get numbers from ID. in order to do this the OTSU threshold is used to grab contour 
from the image some flags are implemented to control if the contour is authenticated or not, these numbers are used as inputs of the CNN model.

### model prediction:

Although some models like AdaBoost, KNN, and Bayes with a dimensional reduction like PCA, and LDA can get a good result in prediction, here the CNN model is used and get 100% accuracy with just a few ID card samples. the model is used with a cross-entropy loss function because we dealing with a classification problems.

<img src="https://user-images.githubusercontent.com/54494078/206301499-babc4d6e-2272-46fb-a5d8-bee076c71a5f.jpg" width="1100" height="500" align = 'center' >

Above picture illustrates the accuracy and loss for each epoch.

### libraries:

The entire process is visualized in HTML with javascript, ajax is used to transfer base64 images from the server which is flask to the client and vice versa.
OpenCV and PIL packages are used for image preprocessing in addition to NumPy and also retinaface packages for detecting faces for better alignment.
In the end, Keras is used for implementing the CNN model.

