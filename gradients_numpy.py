
#--- The Model Training Using Numpy------##
import numpy as np
# f = w * x
# f = 2 * x
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)
w = 0.0
#model predication
def forward(x):
    return w * x
# loss = MSE
def loss(y, y_predicated):
    return((y_predicated-y)**2).mean()
#gradient
# MSE = 1/N * (w*x - y)**2
#dj/dw = 1/N 2x (w*x -y)
def gradient(x,y,y_predicated):
    return np.dot(2 *x, y_predicated-y).mean()
print(f'Predication Before Training: f(5) = {forward(5):.3f}')
#Traning
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
#predication forward pass
     y_pred = forward(X)
     #loss
     l = loss(Y, y_pred)
     #Gradients
     dw = gradient(X,Y,y_pred)
     #Updates Weights
     w -= learning_rate * dw
     if epoch % 2 == 0:
         print(f'epoch {epoch+1}: w = {w:3f}, loss = {l:.8f}')
print(f'Predication after Training: f(5) = {forward(5):.3f}')         



