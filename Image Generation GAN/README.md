# Image Generation with GAN #

The program is implemented in Python 3. The training of the neural netwroks require several hours, around 8 for 25 epochs.
The GAN is composed of two neural networks the Generator and the Discriminator. The purpose of this program to learn from real images to generate new images.
In the result folder it is possible to see the real images on which the GAN is trained and the images generated after each epoch of training.
From the images we can notice, that after each epoch the quality of the image improve. So if you want to obtain better images you can also train the GAN for more than 25 epochs.
This is an example of the generated images after 25 epochs :

![alt text](/Image%20Generation%20GAN/results/fake_samples_epoch_024.png "Example of generated images")

The dataset can be found at the following link:

https://drive.google.com/open?id=1Um4Jfv7d-Nh407Z31MFQpWWtVhfKddOC

Run the 'GAN.py' file to start training the GAN.


