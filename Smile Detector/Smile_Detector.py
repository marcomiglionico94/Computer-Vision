# Face Recognition
# Smile Detector

# Importing Libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

'''
Function for detection:
The detection works with single image from the webcam
Input:
    - gray, balck and white version of the image in which we will detect
    - frame, original version of the image in which we will draw rectangles
Output:
    - original image with rectangles
'''
def detect(gray, frame):
    # Faces tuple
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Region of interest inside the rectangle for gray image
        roi_gray = gray[y:y+h, x:x+y]
        # Region of interest inside the rectangle for color image
        roi_color = frame[y:y+h, x:x+y]
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes: # for each detected eye
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255,0), 2)
        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles: # for each detected smile
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

# Face detection with webcam
    
# Enable the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read() # get the last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to gray scale
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas) # Display output
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # If we type on the keyboard:
        break # We stop the loop.
        
video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.



