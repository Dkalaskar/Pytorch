#Linear Regression Using Pytorch 

###-- The Model Trainig Using Pytorch
# 1 ) Design Model (input, output size, forward pass)
# 2 ) Construct loss and optmizer
# 3 } Training Loop
#  - forward pass : compute predication
#  - backward pass : gradients
#  - update weights

import torch
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0 prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1,noise=20,random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape 

#model

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#traning loop 
num_epoch = 100
for epoch in range(num_epoch):
    #forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    #backward
    loss.backward()
    
    #update
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch +1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
#plot

predicited = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy,predicited, 'b')
plt.show()
       
        
        

