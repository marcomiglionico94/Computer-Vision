# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Define the generator class
        
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__() # We inherit from the nn.Module tools.
        self.main = nn.Sequential(  # We create a meta module of a neural network that will contain a sequence of modules
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # Inversed Convolution Layer
                nn.BatchNorm2d(512), # Apply Batch Normalization
                nn.ReLU(True), # Add Relu to break linearity
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),  # Additional Inversed Convolution Layer
                nn.BatchNorm2d(256), # Apply Batch Normalization
                nn.ReLU(True), # Add Relu to break linearity
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),  # Additional Inversed Convolution Layer
                nn.BatchNorm2d(128), # Apply Batch Normalization
                nn.ReLU(True), # Add Relu to break linearity
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),  # Additional Inversed Convolution Layer
                nn.BatchNorm2d(128), # Apply Batch Normalization
                nn.ReLU(True), # Add Relu to break linearity
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),  # Additional Inversed Convolution Layer
                nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
                )
 
# We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.       
    def forward(self, input):
            output = self.main(input)  # We forward propagate the signal through the whole neural network of the generator defined by self.main.
            return output # Return the output of the neural network


        
# Define the discriminator class

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__() # We inherit from the nn.Module tools.
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias = False), # Convolution layer
                nn.LeakyReLU(0.2, inplace = True), # We apply a Leaky Relu
                nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution.
                nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.
                nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
                nn.Conv2d(128, 256, 4, 2, 1, bias = False), # We add another convolution.
                nn.BatchNorm2d(256), # We normalize again.
                nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
                nn.Conv2d(256, 512, 4, 2, 1, bias = False), # We add another convolution.
                nn.BatchNorm2d(512), # We normalize again.
                nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
                nn.Conv2d(512, 1, 4, 1, 0, bias = False), # We add another convolution.
                nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
                )

# We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output that will be a value between 0 a 1           
        def forward(self, input):
            output = self.main(input)  # We forward propagate the signal through the whole neural network of the generator defined by self.main.
            return output.view(-1) # Return the output of the neural network flattened through the view function
        
        

# Creating the Generator
net_G = Generator() # Generate the neural netowrk object
net_G.apply(weights_init) # Initialize the weights

# Creating the Discriminator
net_D = Discriminator() # Generate the neural netowrk object
net_D.apply(weights_init) # Initialize the weights

# Training the DCGANs
criterion = nn.BCELoss() # We will use Binary Cross Entropy as loss function
optimizerD = optim.Adam(net_D.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We will use Adam optimizer and specify learning rates and betas
optimizerG = optim.Adam(net_G.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We will use Adam optimizer and specify learning rates and betas

# Training process with n epochs
for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        