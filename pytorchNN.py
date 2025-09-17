# This is a simple implementation of shallow 3-4-1 neural network using Pytorch. 

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # hidden layer with ReLU
        x = torch.sigmoid(self.fc2(x))
        return x


model = NeuralNetwork()

x = torch.randn(5, 3)    # batch of 5 samples, each with 3 features
y_true = torch.randint(0, 2, (5, 1)).float()  # random binary labels (0 or 1)

criterion = nn.BCELoss()               # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

y_pred = model(x)                      # shape (5,1)
loss = criterion(y_pred, y_true)       # compute loss
print("Predictions:", y_pred.squeeze().detach().numpy())
print("Loss before backward:", loss.item())

optimizer.zero_grad()   # clear previous gradients
loss.backward()         # compute gradients (dL/dW)
print("\nGradient for fc1 weights:\n", model.fc1.weight.grad)  # check gradients

optimizer.step()        # apply gradient descent update
print("\nLoss after one update step:")
print(criterion(model(x), y_true).item())