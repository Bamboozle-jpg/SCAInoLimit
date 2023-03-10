# Taken from SCAI assignment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Try model with iris data set
iris = load_iris()
irisTrainX, irisTestX, irisTrainY, irisTestY = train_test_split(iris.data, iris.target, test_size=0.3)

# All vars to change in the reinforcement learning

#The following all have to do with number of layers
numHiddenLayers = 0; # Must be in b/t 0 and 10 inclusive
#Layeractivations
layerActivations = [1 for x in range (0, 10)] # Must stay between 1 and 23
activationOptions = [nn.Identity(), nn.ELU(), nn.Hardshrink(), nn.Hardsigmoid(), nn.Hardtanh(), nn.Hardswish(), 
    nn.LeakyReLU(), nn.LogSigmoid(), nn.PReLU(), nn.ReLU(), nn.ReLU6(), nn.RReLU(),
    nn.SELU, nn.CELU(), nn.GELU(), nn.Sigmoid(), nn.SiLU, nn.Mish(), nn.Softplus(), 
    nn.Softshrink(), nn.Softsign(), nn.Tanh(), nn.Tanhshrink(), nn.GLU()]
#Lengths of layers
layerLengths = [5 for x in range(0, 11)];  # Must stay above 1

epochs = 10 #B/t 10 and 40
lossFunction = []
learningRate = 0.0001; # Must be b/t 0.0001 and 0.05
optimNum = 0; # 0-12
confidenceCurrent = 0 # How much to change stuff? IDK

#Ran every time to create the NN model based off of above
def makeAndTestNN():

    activationFunctions = [
        activationOptions[layerActivations[0]], 
        activationOptions[layerActivations[1]], 
        activationOptions[layerActivations[2]], 
        activationOptions[layerActivations[3]], 
        activationOptions[layerActivations[4]], 
        activationOptions[layerActivations[5]], 
        activationOptions[layerActivations[6]], 
        activationOptions[layerActivations[7]], 
        activationOptions[layerActivations[8]], 
        activationOptions[layerActivations[9]]
    ]

    openingLayer = nn.Linear(len(irisTrainX[0]), layerLengths[0])

    layers = [ nn.Linear(layerLengths[x], layerLengths[x+1]) for x in range(0, 10) ]
    
    #For getting the final output (3 options)
    finalActivation = nn.Softmax(3)

    #Turning any layers into identity activations if we want it to be shorter
    for i in range(numHiddenLayers, 10):
        layers[i] = activationOptions[0]
    for i in range(numHiddenLayers, 10):
        activationFunctions[i] = activationOptions[0]

    model = nn.Sequential(
        openingLayer,
        activationFunctions[1],
        layers[1],
        activationFunctions[2],
        layers[2],
        activationFunctions[3],
        layers[3],
        activationFunctions[4],
        layers[4],
        activationFunctions[5],
        layers[5],
        activationFunctions[6],
        layers[6],
        activationFunctions[7],
        layers[7],
        activationFunctions[8],
        layers[8],
        activationFunctions[9],
        layers[9],
        finalActivation
    )

    print(model)

    if (optimNum == 0):
        optimFunc = torch.optim.Adadelta(model.parameters(), lr = learningRate)
    elif (optimNum == 1):
        optimFunc = torch.optim.Adagrad(model.parameters(), lr = learningRate)
    elif (optimNum == 2):
        optimFunc = torch.optim.Adam(model.parameters(), lr = learningRate)
    elif (optimNum == 3):
        optimFunc = torch.optim.AdamW(model.parameters(), lr = learningRate)
    elif (optimNum == 4):
        optimFunc = torch.optim.SparseAdam(model.parameters(), lr = learningRate)
    elif (optimNum == 5):
        optimFunc = torch.optim.Adamax(model.parameters(), lr = learningRate)
    elif (optimNum == 6):
        optimFunc = torch.optim.ASGD(model.parameters(), lr = learningRate)
    elif (optimNum == 7):
        optimFunc = torch.optim.LBFGS(model.parameters(), lr = learningRate)
    elif (optimNum == 8):
        optimFunc = torch.optim.NAdam(model.parameters(), lr = learningRate)
    elif (optimNum == 9):
        optimFunc = torch.optim.RAdam(model.parameters(), lr = learningRate)
    elif (optimNum == 10):
        optimFunc = torch.optim.RMSprop(model.parameters(), lr = learningRate)
    elif (optimNum == 11):
        optimFunc = torch.optim.Rprop(model.parameters(), lr = learningRate)
    elif (optimNum == 11):
        optimFunc = torch.optim.SGD(model.parameters(), lr = learningRate)

    optimizer = optimFunc
makeAndTestNN()