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
                nn.BatchNorm2d(64), # Apply Batch Normalization
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
        
        

if __name__ == '__main__':
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
        for i, data in enumerate(dataloader, 0): # We iterate over the images
            
            # 1st Step: Updating the weights of the neural network of the discriminator
            net_D.zero_grad() # We initialize to 0 the gradients of the Discriminator 
            
            # Training the discriminator with a real image of the dataset
            real, _ = data # We get a real image of the dataset which will be used to train the discriminator, the second argument is the label but we don't need it    
            input = Variable(real) # We put it into a torch Variable
            target = Variable(torch.ones(input.size()[0])) # Target will be a torch Variable of ones
            output = net_D(input) #  We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1)
            errD_real = criterion(output, target) # Calculate the loss error of training of the discriminator with real images
            
            # Training the discriminator with a fake image generated by the generator
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # We make a random vector of noise for the generator
            fake = net_G(noise) # We feed the noise into the genrator to obtain fake images
            target = Variable(torch.zeros(input.size()[0])) # The target in this case will be a tensor full of 0, since 0 correspond to the fake images
            output = net_D(fake.detach())  #  We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1). Detach is used to save memory, we detach the gradient because we don't need it
            errD_fake = criterion(output, target) # Calculate the loss error of training of the discriminator with fake images
            
            # Backpropagating the total error
            errD = errD_real + errD_fake # Total error
            errD.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
            optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.
            
            # 2nd Step: Updating the weights of the neural network of the Generator
            net_G.zero_grad() # We initialize to 0 the gradients of the Generator
            target = Variable(torch.ones(input.size()[0])) # Target will be a torch Variable of ones
            output = net_D(fake)  # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errG = criterion(output, target)  # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
            errG.backward()  # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
            optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.
            
            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
            
            print('[%d/%d][%d/%d] Loss_D: %.4f  Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
            # Let's save the image every 100 steps
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % ("./results"), normalize = True) # Save real image of the mini batch
                fake = net_G(noise) # We get our fake generated images.
                vutils.save_image(fake.data, '%s/fake_samples_epoch%03d.png' % ("./results", epoch), normalize = True) # Save fake generated images of the mini batch
                
                
            
            
            
            