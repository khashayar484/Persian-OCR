# Persian OCR.
this project aim to get national ID code from its images. the main problem is that the national ID can be located anywhere on a paper with irrgular size and scale, so here, first try to locate frame of the card, align it after than cropping the nummber, scale it and then get the ID numbers, after that predict each number by CNN model.

## Table of contents:

the main processes of predicting ID card is mention below.

### Frame detection and alignment:
because images can be located anywhere, first we need to align the frame then crop the ID number based on its location. to do this 
after inverting the image BGR2GRAY with OpenCv, grab the largest contour which is ID card's frame, align the frame which gets from 
the conjuction between edge and the center, this method is more precisly than aligning ID card by a it's face.
<img src="https://user-images.githubusercontent.com/54494078/207532768-998c2398-b4b9-4965-a523-e53b020cbfc8.jpg" width="1000" height="500" align = 'center' >
<img src="https://user-images.githubusercontent.com/54494078/207542106-6011eb79-9895-4ebe-bd67-1904e35a96ab.jpg" width="1000" height="400" align = 'center' >
![OCR_1 (2)](https://user-images.githubusercontent.com/54494078/207532768-998c2398-b4b9-4965-a523-e53b020cbfc8.jpg width="500" height="400" align = 'center')
![OCR_2 (7)](https://user-images.githubusercontent.com/54494078/207542106-6011eb79-9895-4ebe-bd67-1904e35a96ab.jpg width="500" height="400" align = 'center')

above image shows one of the sample of ID card locate on paper. as you can see a right image shows the input image and the left one shows the trim image which shows the frame after scaling and cropping.  
![OCR_3 (1)](https://user-images.githubusercontent.com/54494078/207532858-57130dcc-3545-4270-869a-d44a1c41398d.jpg width="500" height="400" align = 'center')

some images like above cropped before, so here we need to set autentications for cropping images or not, then align the result.

### ID detection:
after grabbing frame and detecting the ID number, we need to get numbers from ID. inorder to do this the OTSU threshold is used to grab contour 
from image some flags is implemented to control if the contour is autenticate or not, these numbers is used as inputs of CNN model.

### model prediction:
although some models like AdaBoost, KNN, Bayes with its dimensional reduction like PCA,LDA can get the good result in prediction, but here the CNN model is used and get the 100% accuracy with just afew ID card samples. the model is used with cross entropy loss function because we dealing with classification problem.

![Capture](https://user-images.githubusercontent.com/54494078/206301499-babc4d6e-2272-46fb-a5d8-bee076c71a5f.jpg width="500" height="400" align = 'center')

above picture illustrates the accuracy and loss for each epoch.

### libraries:
the entire processes are visualized in html with javascript, ajax is used to transfer base64 image from server which is flask to client and viceversa.
OpenCv and PIL packages is used for image preprocessing addition to numpy and also retinaface package for detecting face for better alignment.
at the end keras is used for implementing the CNN model.

