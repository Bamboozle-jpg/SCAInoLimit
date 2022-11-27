import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import random_split, TensorDataset, DataLoader

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

    # Function for training the model
    def trainn(self, numEpochs, trainLoader, validateLoader):
        lossFn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        bestAccuracy = 0.0
        
        print("Training with", numEpochs, "epochs...")
        for epoch in range(1, numEpochs + 1):
            # For each epoch resets epoch vars
            runningTrainingLoss = 0.0
            runningAccuracy = 0.0
            runningValLoss = 0.0
            total = 0

            # Actually trains
            for data in trainLoader:
                inputs, outputs = data
                outputs = outputs.long()
                # Zero param gradients
                optimizer.zero_grad()
                predictedOutputs = self.forward(inputs)
                # Sets up and uses backpropogation to optimize
                trainLoss = lossFn(predictedOutputs, outputs[:, 0])
                trainLoss.backward()
                optimizer.step()
                runningTrainingLoss += trainLoss.item()

            trainLossValue = runningTrainingLoss/len(trainLoader)

            # Validation (AKA Figure out which model change was the best)
            with torch.no_grad():
                self.eval()
                for data in validateLoader:
                    inputs, outputs = data
                    outputs = outputs.long()
                    # Gets values for loss
                    predictedOutputs = self(inputs)
                    valLoss = lossFn(predictedOutputs, outputs[:, 0])
                    # Highest value will be our prediction
                    _, predicted = torch.max(predictedOutputs, 1)
                    runningValLoss += valLoss.item()
                    total += outputs.size(0)
                    runningAccuracy += (predicted == outputs).sum().item()
            
            # Calculate Validation Loss Val
            valLossValue = runningValLoss/len(validateLoader)
            # Accuracy = num of correct predictions in validation batch / total predictions done
            accuracy = (100 * runningAccuracy / total)

            # Save model if accuracy is best
            if accuracy > bestAccuracy:
                self.saveModel("testModel.pth")
                bestAccuracy = accuracy

            # Print current Epoch stats
            print("Completed training for epoch :", epoch, 'Training Loss is %.4f' %trainLossValue, 'Validation Loss is: %.4f' %valLossValue, 'Accuracy is %d %%' % (accuracy))


# Grabs data and turns it into usable form:

# Grabs raw list of dictionaries
everythingDict=[*csv.DictReader(open('finalData.csv'))]
actualDict = {}
actualDict['solov'] = []
actualDict['height'] = []
actualDict['inundate'] = []

# Turns list of dictionaries into dictionary of lists
for i in range(0, len(everythingDict)):
    actualDict['solov'].append(float(everythingDict[i]['solovievIdentity']))
    actualDict['height'].append(float(everythingDict[i]['waveHeight']))
    actualDict['inundate'].append(float(everythingDict[i]['horizontalInundation']))

# Creates Pandas dataframe for input and output
df = pd.DataFrame(actualDict)
input = df.loc[:, ['height', 'inundate']]
output = df.loc[:, ['solov']]

# Turns pandas dataframes into tensors and Tensor Dataset
input = torch.Tensor(input.to_numpy())
output = torch.Tensor(output.to_numpy())
data = TensorDataset(input, output)

# Split into a training, validation and testing set
trainBatchSize = 10
testSplit = int(len(input)*0.25)
print(testSplit)
trainSplit = int(len(input)*0.6)
print(trainSplit)
validateSplit = len(input) - trainSplit - testSplit
print(validateSplit)
print(len(input))
trainSet, validateSet, testSet = random_split(data, [trainSplit, validateSplit, testSplit])

# Get data in loadable form to go into model
trainLoader = DataLoader(trainSet, batch_size=trainBatchSize, shuffle=True)
validateLoader = DataLoader(validateSet, batch_size=1)
testLoader = DataLoader(testSet, batch_size=1)

# Sets input and output size for future models
inputSize = list(input.shape)[1]
solovs = []
for i in actualDict['solov']:
    if not i in solovs:
        solovs.append(i)
print(solovs)
outputSize = len(solovs)


# TRAINING AND TESTING MODEL!!!

# Actually put it into the model
# For loading current one
waveModel = testModel.loadModel(inputSize, outputSize, "testModel.pth")
# For creating new one
# waveModel = testModel(inputSize, outputSize)
print("input size :", inputSize)
print("output size :", outputSize)
waveModel.trainn(10, trainLoader, validateLoader)
