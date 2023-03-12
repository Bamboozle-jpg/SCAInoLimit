# Taken from SCAI assignment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Try model with iris data set
iris = load_iris()
parameters = torch.tensor(iris.data)
parameters = parameters.to(torch.float32)

target = torch.tensor(iris.target)
target = target.to(torch.float32)
target = target[:, None]

irisTrainX, irisTestX, irisTrainY, irisTestY = train_test_split(parameters, target, test_size=0.3)

# All vars to change in the reinforcement learning

#The following all have to do with number of layers
numHiddenLayers = 10; # Must be in b/t 0 and 10 inclusive
#Layeractivations
layerIndices = [1 for x in range (0, 10)] # Must stay between 1 and 22
activationOptions = [nn.Identity(), nn.ELU(), nn.Hardshrink(), nn.Hardsigmoid(), nn.Hardtanh(), nn.Hardswish(), 
    nn.LeakyReLU(), nn.LogSigmoid(), nn.PReLU(), nn.ReLU(), nn.ReLU6(), nn.RReLU(),
    nn.SELU(), nn.CELU(), nn.GELU(), nn.Sigmoid(), nn.SiLU(), nn.Mish(), nn.Softplus(), 
    nn.Softshrink(), nn.Softsign(), nn.Tanh(), nn.Tanhshrink(), nn.GLU()]
#Lengths of layers
layerLengths = [5 for x in range(0, 10)];  # Must stay above 1

lossIndex = 1 # b/t 0 and 21
lossOptions = [nn.L1Loss(), nn.MSELoss(), nn.CrossEntropyLoss(), nn.CTCLoss(), nn.NLLLoss(), 
    nn.PoissonNLLLoss(), nn.GaussianNLLLoss(), nn.KLDivLoss(), nn.BCELoss(), 
    nn.BCEWithLogitsLoss(), nn.MarginRankingLoss(), nn.HingeEmbeddingLoss(), 
    nn.MultiLabelMarginLoss(), nn.HuberLoss(), nn.SmoothL1Loss(), nn.SoftMarginLoss(), 
    nn.MultiLabelSoftMarginLoss(), nn.CosineEmbeddingLoss(), nn.MultiMarginLoss(), 
    nn.TripletMarginLoss(), nn.TripletMarginWithDistanceLoss()]

epochs = 300 # b/t 10 and 300
learningRate = 0.001; # Must be b/t 0.001 and 0.1
optimNum = 0; # 0-12

#Ran every time to create the NN model based off of above
def makeAndTestNN():

    activationFunctions = [
        activationOptions[layerIndices[0]], 
        activationOptions[layerIndices[1]], 
        activationOptions[layerIndices[2]], 
        activationOptions[layerIndices[3]], 
        activationOptions[layerIndices[4]], 
        activationOptions[layerIndices[5]], 
        activationOptions[layerIndices[6]], 
        activationOptions[layerIndices[7]], 
        activationOptions[layerIndices[8]], 
        activationOptions[layerIndices[9]]
    ]

    openingLayer = nn.Linear(len(irisTrainX[0]), layerLengths[0])
    layers = [ nn.Linear(layerLengths[x], layerLengths[x+1]) for x in range(0, 9) ]
    finalLayer = nn.Linear(layerLengths[9], 3)
    
    #For getting the final output (3 options)
    finalActivation = nn.Softmax(dim=1)

    #Turning any layers into identity activations if we want it to be shorter
    for i in range(numHiddenLayers, 9):
        layers[i] = activationOptions[0]
    for i in range(numHiddenLayers, 10):
        activationFunctions[i] = activationOptions[0]

    model = nn.Sequential(
        openingLayer,
        activationFunctions[0],
        layers[0],
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
        finalLayer,
        finalActivation
    )

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

    losses = []
    for e in range(epochs):
        predY = model(irisTrainX)
        loss = lossOptions[lossIndex](predY, irisTrainY)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimFunc.step()

    # PredY = predicted Y values
    predY = model(irisTestX)


    #Takes model and tries it with test values to see how well it did
    numRight = 0
    numWrong = 0

    for index in range(0,len(predY)):
        currentMax = 0
        currentChoice = 0
        for option in range(0,3):
            if predY[index][option] > currentMax:
                currentMax = predY[index][option]
                currentChoice = option
        
        if currentChoice == irisTestY[index]:
            numRight += 1
        else:
            numWrong += 1

    return (numRight/(numWrong+numRight))

# Determines which of the bests should be used as base
def bestIndex():
    test = random.randrange(0, 6)
    if test > 2:
        choice = 0
    elif test > 0:
        choice = 1
    else:
        choice = 2
    return choice

# Decides if optimizer should be changed
def optimChanger(frequency = 7):
    test = random.randrange(0, 10)
    if test > frequency:
        return bestOptimNums[bestIndex()]
    else:
        return random.randrange(0,13)

# Decides if activation should be changed
def activationChanger(frequency = 7):
    test = random.randrange(0, 10)
    if test > frequency:
        return bestLayerIndices[bestIndex()][5]
    else:
        return random.randrange(1,24)

# Things to change : 
#   Epochs 10-300 int
epochs = 300
#   Num layers 0-10 int
numHiddenLayers = 10
#   Length of layers 1-? length of 11
layerLengths = [10 for x in range(0, 10)] 
#   Learning Rate 0.1-0.001 int
learningRate = 0.01
#   optimizer 0-12 int 
optimNum = 0
#   activation options 1-23, length of 10
layerIndices = [7 for x in range (0, 10)]

# Tracks how well it's doing overtime
improvement = [0] * 50

# Best numbers from current run in temp, then moved to best for next run
bestEpochs = [epochs, epochs, epochs]
bestTempEpochs = [epochs, epochs, epochs]

bestNumHiddenLayers = [numHiddenLayers, numHiddenLayers, numHiddenLayers]
bestTempNumHiddenLayers = [numHiddenLayers, numHiddenLayers, numHiddenLayers]

bestlearningRates = [learningRate, learningRate, learningRate]
bestTemplearningRates = [learningRate, learningRate, learningRate]

bestOptimNums = [optimNum, optimNum, optimNum]
bestTempOptimNums = [optimNum, optimNum, optimNum]

bestLayerIndices = [[0]*10]*3
for place in range(len(bestLayerIndices)):
    for layer in range(len(layerIndices)):
        bestLayerIndices[place][layer] = layerIndices[layer]
bestTempLayerIndices = [[0]*10]*3
for place in range(len(bestTempLayerIndices)):
    for layer in range(len(layerIndices)):
        bestTempLayerIndices[place][layer] = layerIndices[layer]

bestLayerLengths = [[0]*10]*3
for place in range(len(bestLayerLengths)):
    for layer in range(len(layerLengths)):
        bestLayerLengths[place][layer] = layerLengths[layer]
bestTempLayerLengths = [[0]*10]*3
for place in range(len(bestTempLayerLengths)):
    for layer in range(len(layerLengths)):
        bestTempLayerLengths[place][layer] = layerLengths[layer]

bestResults = [0, 0, 0]

numzeros = 0
numones = 0
numtwos = 2

for j in range(0, 50):
    # Resets best results so it fills the best lists with all new values
    improvement[j] = (bestResults[0] + bestResults[1] + bestResults[2])/3
    bestResults = [0, 0, 0]
    for place in range(len(bestTempEpochs)):
        bestEpochs[place] = bestTempEpochs[place]
        bestNumHiddenLayers[place] = bestTempNumHiddenLayers[place]
        for layer in range(len(bestLayerLengths[0])):
            bestLayerLengths[place][layer] = bestTempLayerLengths[place][layer]
        bestlearningRates[place] = bestTemplearningRates[place]
        bestOptimNums[place] = bestTempOptimNums[place]
        for layer in range(len(bestLayerIndices[0])):
            bestLayerIndices[place][layer] = bestTempLayerIndices[place][layer]

    # 30 tries with small changes to the bests
    for i in range(0, 30):
        # Gets a weighted random of the bests, and then adds a small change to it
        # Changes how many epochs it uses
        epochs = max(10, min(300, bestEpochs[bestIndex()] + random.randint(-20, 20)))

        # Changes how many hidden layers there are
        numHiddenLayers = max(0, min(10, bestNumHiddenLayers[bestIndex()] + random.randint(-2, 2)))

        # # Changes which optimizer it uses
        # optimNum = max(0, min(12, optimChanger()))

        # Changes the learning rate
        learningRate = max(0.001 , min(0.1, bestlearningRates[bestIndex()] * (10**(random.random()-.5))))

        # # Changes the layer activations
        # activation = max(1, min(22, activationChanger()))
        # for layer in range(0, len(layerIndices)):
        #     layerIndices[layer] = activation
        
        # # Changes how long each layer is
        for layer in range(0, len(layerLengths)):
            layerLengths[layer] = max(0, min(10, bestLayerLengths[bestIndex()][layer] + random.randint(-2, 2)))

        # Runs the NN with current Settings
        try:
            results = makeAndTestNN()
        except:
            results = 0

        print("")
        print("trial " + str(i+1) + "/50 of " + str(j+1) + "/50")
        print("epochs : " + str(epochs))
        print("numHiddenLayers : " + str(numHiddenLayers))
        print("optimNums : " + str(optimNum))
        print("learning rate : " + str(learningRate))
        print("Layer activations : " + str(layerIndices))
        print("Layer Lengths : " + str(layerLengths))
        print("Results : " + str(results))
        print("Current Bests : " + str(bestResults))

        # Puts the results and params in the right place if it's better than previous ones
        if results > bestResults[0]:
            placerIndex = 0
        elif results > bestResults[1]:
            placerIndex = 1
        elif results > bestResults[2]:
            placerIndex = 2
        else:
            placerIndex = 3

        # Actually places the values into the best lists
        if placerIndex != 3:
            bestResults[placerIndex] = results
            bestTempEpochs[placerIndex] = epochs
            bestTempNumHiddenLayers[placerIndex] = numHiddenLayers
            bestTempOptimNums[placerIndex] = optimNum
            bestTemplearningRates[placerIndex] = learningRate
            for layer in range(len(layerIndices)):
                bestTempLayerIndices[placerIndex][layer] = layerIndices[layer]
            for layer in range(len(layerLengths)):
                bestTempLayerLengths[placerIndex][layer] = layerLengths[layer]

    # Now does very drastic and big changes
    for i in range(0, 20):
        
        # Gets a weighted random of the bests, and then adds a small change to it
        # Changes how many epochs it uses
        epochs = max(10, min(300, bestEpochs[bestIndex()] + random.randint(-75, 75)))

        # Changes how many hidden layers there are
        numHiddenLayers = max(0, min(10, bestNumHiddenLayers[bestIndex()] + random.randint(-4, 4)))

        # # Changes which optimizer it uses
        # optimNum = max(0, min(12, optimChanger(4)))

        # Changes the learning rate
        learningRate = max(0.001 , min(0.1, bestlearningRates[bestIndex()] * (10**((random.random()-.5)*2))))

        # # Changes the layer activations
        # activation = max(1, min(22, activationChanger(4)))
        # for layer in range(0, len(layerIndices)):
        #     layerIndices[layer] = activation
        
        # Changes how long each layer is
        for layer in range(0, len(layerLengths)):
            layerLengths[layer] = max(0, min(10, bestLayerLengths[bestIndex()][layer] + random.randint(-4, 4)))

        # Runs the NN with current Settings
        try:
            results = makeAndTestNN()
        except:
            results = 0
        print("")
        print("trial " + str(i+30) + "/50 of " + str(j+1) + "/50")
        print("epochs : " + str(epochs))
        print("numHiddenLayers : " + str(numHiddenLayers))
        print("optimNums : " + str(optimNum))
        print("learning rate : " + str(learningRate))
        print("Layer activations : " + str(layerIndices))
        print("Layer Lengths : " + str(layerLengths))
        print("Results : " + str(results))
        print("Current Bests : " + str(bestResults))

        # Puts the results and params in the right place if it's better than previous ones
        if results > bestResults[0]:
            placerIndex = 0
        elif results > bestResults[1]:
            placerIndex = 1
        elif results > bestResults[2]:
            placerIndex = 2
        else:
            placerIndex = 3

        if placerIndex != 3:
            bestResults[placerIndex] = results
            bestTempEpochs[placerIndex] = epochs
            bestTempNumHiddenLayers[placerIndex] = numHiddenLayers
            bestTempOptimNums[placerIndex] = optimNum
            bestTemplearningRates[placerIndex] = learningRate
            for layer in range(len(layerIndices)):
                bestTempLayerIndices[placerIndex][layer] = layerIndices[layer]
            for layer in range(len(layerLengths)):
                bestTempLayerLengths[placerIndex][layer] = layerLengths[layer]

print(improvement)
        



