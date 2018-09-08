# Face Recognition #
The program is implemented in Python 3 using openCV2 library and it can recognize face and eyes in real time from the webcam.
It use 2 haar cascades for face and eyes detection:

*'haarcascade_frontalface_default.xml'
*'haarcascade_eye.xml'

To run the program you just need to have OpenCV2 installed and Python 3.
To close the webcam press 'q' and in case the webcam doesn't open try to change in line 39 the parameter of videoCapture(0) to 1 because it can depends on the webcam settings.

 ![alt text](Screenshot/FaceRecognition.png )
