# PyTorch installation: https://pytorch.org/get-started/locally/

# Please attend the office hours of TA or instructor if you have any question about PyTorch installation.

# The code for neural network training on CIFAR-10 data set is in the file Assignment8.py, you need to add one or several convolutional layers to improve the accuracy of the network on the classification of CIFAR-10 data set.

# TO DO:

# 1. Construct the architecture of convolutional layers in the code (50 points)

# 2. Include the forward propagation of convolutional layers in the code (50 points)

# 3. (Extra) Tune the hyperparameters to achieve a classification accuracy higher than 60% (Extra 10 points)



# Please submit the code and the screenshot of "Accuracy of the network on the 10000 test images" on Blackboard.

# Assignment8.py

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':

    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            #############################################################################
            #  TO DO: 50 points                                                         #
            # - Construct the architecture of convolutional layers                      #
            #############################################################################


            # self.fc1 = nn.Linear(3072, 120)
            # self.fc2 = nn.Linear(120, 84)
            # self.fc3 = nn.Linear(84, 10)

            self.conv1 = torch.nn.Conv2d(3, 16, 4)
            self.pool1 = torch.nn.MaxPool2d(2)
            self.conv2 = torch.nn.Conv2d(16, 32, 4)
            self.pool2 = torch.nn.MaxPool2d(2)
            self.conv3 = torch.nn.Conv2d(32, 64, 4)
            self.pool3 = torch.nn.MaxPool2d(2)

            # self.fc1 = nn.Linear(64, 10)
            # self.fc2 = nn.Linear(64, 32)
            # self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            #############################################################################
            #  TO DO: 50 points                                                         #
            # - Include the forward propagation of convolutional layers                 #
            #############################################################################

            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = (self.fc1(x))

            return x


    net = Net()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
