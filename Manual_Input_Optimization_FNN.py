# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:16:36 2023

@author: Korisnik
"""




#%% Libraries

import torch
import torch.nn as nn
import torch.optim as optim

#%% Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(2, 3)

        self.output_layer = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden_layer(x))
        x = (self.output_layer(x))
        return x

net = Net()

#%% Architecture and weights and biases

print(f"network topology: {net}")

print(f"hidden layer weight: {net.hidden_layer.weight}")
print(f"hidden layer bias: {net.hidden_layer.bias}")
print(f"output layer weight: {net.output_layer.weight}")
print(f"output layer bias: {net.output_layer.bias}")



#%% Run input data forward through network

input_data = torch.tensor([3.0, 0.0])
output = net(input_data)
print(output.data)

#%% Backpropagate gradient

target = torch.tensor([0.75, 0.5])
criterion = nn.MSELoss()  # Our loss function
loss = criterion(output, target) # Calculates loss for out output vs target
net.zero_grad() # Set gradiant of all paramaters with requires_grad=True to 0, so it does not accumulates over different runs
loss.backward() # Calculates and stores dloss/dparam for each param as param.grad

#%%

print(net.hidden_layer.bias.grad)


#%% Update weights and biases

optimizer = optim.SGD(net.parameters(), lr=0.1) # Defining optimization method and learning rate lr
optimizer.step() # Updates params as param += - lr * param.grad

print(f"updated_w_l1 = {round(net.hidden_layer.weight.item(), 4)}")
print(f"updated_b_l1 = {round(net.hidden_layer.bias.item(), 4)}")
print(f"updated_w_l2 = {round(net.output_layer.weight.item(), 4)}")
print(f"updated_b_l2 = {round(net.output_layer.bias.item(), 4)}")

output = net(input_data)
print(f"updated_a_l2 = {round(output.item(), 4)}")


#%% Try to optimize input such that output is target without changing parameters

def delta_input(w1, b1_grad, learning_rate = 0.5):
    input_grad = torch.transpose(w1, 0, 1) @ b1_grad
    return( - learning_rate * input_grad )



input_data = torch.tensor([1.0, 1.0])
target = torch.tensor([1.0])

criterion = nn.MSELoss()

for i in range(10000):
    
    output = net(input_data)
    loss = criterion(output, target)
    
    net.zero_grad()
    loss.backward()
    
    input_data += delta_input(net.hidden_layer.weight.data, net.hidden_layer.bias.grad.data)

#%%

print( net(input_data) )
print( input_data )



