# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:39:23 2023

@author: Korisnik
"""




#%% Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

#%% Define transformation on data

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081)) ])

#%% Downloading data

trainset = datasets.MNIST('MNIST_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('MNIST_VALSET', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#%% Getting acquainted with the data

data_enumerate = enumerate(trainloader)
idx, (images, labels) = next(data_enumerate)

print(images.shape)
print(labels.shape)

#%% Seeing is believing

plt.imshow(images[2].numpy().squeeze(), cmap='gray_r')

#%% Defining the model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 7)
        self.conv2 = nn.Conv2d(10, 10, 5)
        
        self.fc1 = nn.Linear(10*18*18, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10*18*18)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return(x)
    
net = Net()

#%%

print(net)

#%%

net(images[0])

#%%

criterion = nn.NLLLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)

epochs = 4

for epoch in range(epochs):
    
    for inputs, targets in trainloader:
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
#%% Checking how good the model is

correct_count, all_count = 0, 0
for images, targets in valloader:
  for i in range(len(targets)):
    image = images[i]
    with torch.no_grad():
        logps = net(image)

    
    prob = torch.exp(logps)
    prob = list(prob.numpy()[0])
    pred_label = prob.index(max(prob))
    true_label = targets.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1


print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count*100))


#%% Try to optimize input such that certain neuron is activated the most

x = torch.nn.Parameter(torch.rand(1, 28, 28), requires_grad=True)

optimizer = optim.SGD([x], lr = 0.1, momentum = 0.5) # Be careful, if lambd is big lr needs to be small, otherwise it becomes chaotic and does not optimize

def regularization(x, lambd = 1): # Since activation is ReLU, it is unbounded. Sometimes it helps, some times results stay the same.
    return( lambd * torch.sqrt(torch.sum(x**2)) )

activation = {}
def get_activation(name):
    def hook(model, inputs, output):
        activation[name] = output
    return hook


model_children = list(net.children())


hook = model_children[1].register_forward_hook(get_activation('cnn_1'))

for i in range(500):
    net(x)
    #loss = - activation['cnn_1'][0, 9] + regularization(x) # Linear layers
    loss = - regularization(activation['cnn_1'][9, :, :]) + regularization(x) # Convolutional layers
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# In case for loop fails, remember we do not want hook to stay

hook.remove()

#%%

plt.imshow(x.data.numpy().squeeze(), cmap='gray_r')



