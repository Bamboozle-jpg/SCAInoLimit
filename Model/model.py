import torch

class testModel(torch.nn.Module):
    def __init__(self):
        super(testModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = testModel()

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:', x)

print('\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\nOutput:')
print(y)

x = torch.rand(1, 100)
exModel = testModel()
y = exModel.forward(x)
print(y)