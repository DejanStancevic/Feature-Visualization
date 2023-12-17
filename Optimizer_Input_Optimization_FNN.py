# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:54:39 2023

@author: Korisnik
"""




#%% Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%% Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(2, 3)

        self.output_layer = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = (self.output_layer(x))
        return x

net = Net()

#%% 

for param in net.parameters():
    print(param)
    

#%% Architecture and weights and biases

print(f"network topology: {net}")

print(f"hidden layer weight: {net.hidden_layer.weight}")
print(f"hidden layer bias: {net.hidden_layer.bias}")
print(f"output layer weight: {net.output_layer.weight}")
print(f"output layer bias: {net.output_layer.bias}")



#%% Run input data forward through network

input_data = torch.tensor([1.0, 1.0])
output = net(input_data)
print(output.data)

#%% Try to optimize input such that output is target without changing parameters

input_data = torch.nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)

target = torch.tensor([-0.8])

criterion = nn.MSELoss()

optimizer = optim.SGD([input_data], lr = 0.5)

for i in range(1000):
    
    output = net(input_data)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    
#%% Check

print( net(input_data) )
print( input_data )

#%% Try to optimize input such that certain neuron is activated the most

x = torch.nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)

optimizer = optim.SGD([x], lr = 0.1) # Be careful, if lambd is big lr needs to be small, otherwise it becomes chaotic and does not optimize

def regularization(x, lambd = 1): # Since activation is ReLU, it is unbounded. Hence, we need to insure input does not grow to extremes
    return( lambd * torch.sum(x**2) )


Hidden_Activation = []


hook = net.hidden_layer.register_forward_hook(lambda model, ins, outs: Hidden_Activation.append(outs[0]))


for i in range(100):
    output = net(x)
    loss = - Hidden_Activation[-1] + regularization(x)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
hook.remove()
    
#%%

print(Hidden_Activation[0], Hidden_Activation[-1])
print( x )



















