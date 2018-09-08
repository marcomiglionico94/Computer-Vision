# -*- coding: utf-8 -*-

# Importing libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Function used for detection
def detect(frame, net, transform):
    '''Detect the object inside an image(frame).

        Args:
            frame: single image extracted from the video
            net: ssd neural network
            transform:  transformation to be applied on the images so that it can have the right format and dimension for the neural network

        Return:
             frame: return the original frame with the detector rectangle and the label around the detected object
             
    '''
     # We get the height and the width of the frame.
    height, width = frame.shape[:2] # We take only the first 2 arguments, because the 3rd is the number of channel(3 = color image, 1 = black and white)
    frame_transformed = transform(frame)[0] # Take only the first element that is the transformed frame
    x = torch.from_numpy(frame_transformed).permute(2, 0, 1) # Convert numpy array into a torch tensor, and change order of color from RBG to GRB, because the SSD was trained under this convention
    x = Variable(x.unsqueeze(0)) # The NN accept only batches, so we use unsqueeze function to add a fake dimension for the batch. 0 is the index of the first dimension
    
    # We already have a pre-trained SSD able to recognize from 30 to 40 objects
    y = net(x) # Feed our torch variable into the pre-trained neural network and we get the output y
    detections = y.data # Detection tensor that contains the data of the output
    
    # Normalize the scaled values of the position of the detected object between 0 and 1
    scale = torch.Tensor([width, height, width, height]) # The first two (width,height) correspond to upper left corner of the rectangle, the other 2 to the lower right corner of the rectangle.
    
    # detections = [batch, number of classes detected, number of occurrence of the class, (score, x0, Y0, x1, Y1)] ->(x0,Y0,x1 and Y1 are coordinates)
    for i in range(detections.size(1)): # For each class
        j = 0
        while detections[0, i, j, 0] >= 0.6: # If the score of the occurrence j of class i is greater than 0.6
            point = (detections[0, i, j, 1:] * scale).numpy() # Save the coordinates of the rectangle scaled between 0 and 1 in a numpy array since openCV works with np arrays
            cv2.rectangle(frame, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), (255,0,0), 2) # Draw the rectangle using the 4 points
            cv2.putText(frame, labelmap[i-1],  (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 255), 2, cv2.LINE_AA) # Display Label
            j += 1 # Increment j
            
    return frame

# Creating the SSD Neural Network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.    

# Object Detection on a video
reader = imageio.get_reader('homework3.mp4') # Open the video
frequence_fps = reader.get_meta_data()['fps'] # Get the fps of the video
writer = imageio.get_writer('output3.mp4', fps = frequence_fps) # Create output video with same fps
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # Call the detect function
    writer.append_data(frame) # Add the frame to the output video
    print(i) # Print the processed frame
writer.close() # Close the process that create the output video
    
    
    