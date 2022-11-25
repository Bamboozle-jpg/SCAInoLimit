import torch

class testModel(torch.nn.Module):
    # Initialize model
    def __init__(self, inputSize, outputSize):
        super(testModel, self).__init__()

        self.linear1 = torch.nn.Linear(inputSize, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.softmax = torch.nn.Softmax()
        self.linear3 = torch.nn.Linear(200, outputSize)

    # Send a tensor through the model
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
    
    # Saves model to file
    def saveModel(self, name):
        path = "./" + name 
        torch.save(self.state_dict(), path)

    # Loads model from file
    def loadModel(inputSize, outputSize, path):
        model = testModel(inputSize, outputSize)
        model.load_state_dict(torch.load("./" + path))
        model.eval()
        return model

# Setup Linear transformation and tensor to test it with
lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:', x)
print('\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

# Testing a linear transformation
y = lin(x)
print('\nOutput:')
print(y)

# Create random tensor
x = torch.rand(1, 100)

# Creates model and tells it to use cpu
exModel = testModel(100, 1)
device = torch.device("cpu")
exModel.to(device)

# Tests Model
y = exModel.forward(x)
print(y)

# Saves Model
exModel.saveModel("testModel.pth")

#Loads model and tests it
model2 = testModel.loadModel(100, 1, "testModel.pth")
y = model2.forward(x)
print(y)

# Loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.0001)