import numpy as np
import pandas as pd
from random import randint

# Load dataset from a csv file
def loadDataFromTextFile(filePath, numberOfData=-1, inputSize=2, labelSize=1):
    print(f"\nLoading dataset from file \"{filePath}\"")
    data = pd.read_csv(filePath, header=None)

    if numberOfData != -1:
        data = data.iloc[:numberOfData]

    data = data.values

    inputData = data[:, :inputSize]
    labels = data[:, inputSize:inputSize + labelSize]
    
    print("Dataset loaded.")
    return np.array(inputData), np.array(labels)

# Generate XOR gate dataset
def generateExampleDataset(numberOfData):
    inputData = []
    labels = []

    for i in range(numberOfData):
        xRandom = randint(0, 1)
        yRandom = randint(0, 1)

        randomData = [xRandom, yRandom]
        result = xRandom ^ yRandom
        
        inputData.append(randomData)
        labels.append(result)

    return inputData, labels
