import numpy as np
import json
import tools

class NeuralNetwork:

    # Constuctor of the class
    def __init__(self, inputLayerNeurons=None, hiddenLayersNeurons=None, outputLayerNeurons=None, modelFile=None, activationFunctions=None):
        # Setting the fields of the class based on the parameters
        self.inputLayerNeurons = inputLayerNeurons
        self.hiddenLayersNeurons = hiddenLayersNeurons or []
        self.outputLayerNeurons = outputLayerNeurons
        self.activationFunctions = activationFunctions

        # Validating the parameters of the class
        self.checkInitialization(inputLayerNeurons, hiddenLayersNeurons, outputLayerNeurons, modelFile, activationFunctions)

        # Setting default parameters for the neural network
        self.epochs = 500
        self.batchSize = 32
        self.learningRate = 0.0001
        self.threshold = 1e-6

        # Initializing weights and biases fields
        self.weights = []
        self.biases = []

        # Checking if the neural network is created to load an already trained model or to train a new one
        if modelFile:
            self.loadModel(modelFile)
        else:
            layersNeuronsSize = self.hiddenLayersNeurons + [self.outputLayerNeurons]
            self.setupWeightsAndBiases(layersNeuronsSize)

    # Validating if the parameters of the class are correct based on the load model or train a new one
    def checkInitialization(self, inputLayerNeurons, hiddenLayersNeurons, outputLayerNeurons, modelFile, activationFunctions):
        # Checking if all the parameters are None and raise an error
        if all(param is None for param in [inputLayerNeurons, hiddenLayersNeurons, outputLayerNeurons, modelFile, activationFunctions]):
            raise ValueError(
                "Initialization Error: All parameters are None. You must provide either a modelFile "
                "to load an existing model OR provide the layer configuration (inputLayerNeurons, "
                "hiddenLayersNeurons, outputLayerNeurons, activationFunctions) to build a new model."
            )

        # Checking if all the parameters have a value
        if all(param is not None for param in [inputLayerNeurons, hiddenLayersNeurons, outputLayerNeurons, modelFile, activationFunctions]):
            raise ValueError(
                "Initialization Error: Conflicting parameters provided. You must provide either a modelFile "
                "OR the explicit layer parameters, but not both."
            )

        # Checking for each layer if the user set an activation function
        expectedActivations = len(self.hiddenLayersNeurons) + 1
        if activationFunctions is None or len(activationFunctions) != expectedActivations:
            received = 0 if activationFunctions is None else len(activationFunctions)
            raise ValueError(
                f"Initialization Error: Mismatch in activation functions. Expected {expectedActivations} activation "
                f"function(s) (one for each hidden layer and one for the output layer), but received {received}."
            )

    # Validating if the input size and labels are correct based on the neural network
    def correctParametersChecking(self, input, labels):
        if len(input[0]) != self.inputLayerNeurons:
            raise ValueError(
                f"Parameter Error: Input size mismatch. Expected input size {self.inputLayerNeurons} "
                f"but received {len(input[0])} for each input."
            )

        if len(labels[0]) != self.outputLayerNeurons:
            raise ValueError(
                f"Parameter Error: Output size mismatch. Expected output size {self.outputLayerNeurons} "
                f"but received {len(labels[0])}."
            )

    # Setting up random weights and biases for the new neural network
    def setupWeightsAndBiases(self, layersNeuronsSize):
        print("\nSetting up the weights and biases of the neural network.")
        inputsNumber = self.inputLayerNeurons

        for neuronsNumber in layersNeuronsSize:
            weightMatrix = np.random.rand(inputsNumber, neuronsNumber) - 0.5
            biasMatrix = np.random.rand(1, neuronsNumber) - 0.5

            self.weights.append(weightMatrix)
            self.biases.append(biasMatrix)

            inputsNumber = neuronsNumber

        print("Setup completed.")

    # Setters for each parameter
    def setEpochs(self, epochs):
        self.epochs = epochs

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setThreshold(self, threshold):
        self.threshold = threshold

    # Method that implements the training of the neural network
    def train(self, trainingData, trainingLabels):
        print("\nStart training the neural network.")
        self.correctParametersChecking(trainingData, trainingLabels)

        numberSamples = trainingData.shape[0]
        for epoch in range(self.epochs):
            indices = np.arange(numberSamples)
            np.random.shuffle(indices)
            trainingDataShuffled = trainingData[indices]
            trainingLabelsShuffled = trainingLabels[indices]

            totalEpochLoss = 0.0
            numberBatches = 0

            for batchStart in range(0, numberSamples, self.batchSize):
                batchEnd = batchStart + self.batchSize
                batchData = trainingDataShuffled[batchStart:batchEnd]
                batchLabels = trainingLabelsShuffled[batchStart:batchEnd]

                activations, preActivations = self.feedForward(batchData)
                predictions = activations[-1]
                batchLoss = np.mean((predictions - batchLabels) ** 2)
                totalEpochLoss += batchLoss
                numberBatches += 1

                self.backpropagation(activations, preActivations, batchLabels)

            averageLoss = totalEpochLoss / numberBatches
            print(f"Epoch {epoch+1}/{self.epochs} - Error: {averageLoss:.6f}")
        
        print("Training completed.")

    # Evaluation of the trained model
    def evaluation(self, testingData, testingLabels):
        print("\nStart evaluating the neural network.")
        self.correctParametersChecking(testingData, testingLabels)

        predictions = []
        for i in range(testingData.shape[0]):
            sample = testingData[i].reshape(1, -1)
            activations, _ = self.feedForward(sample)
            output = activations[-1]

            predicted_label = 1 if output[0, 0] > 0.5 else 0
            predictions.append(predicted_label)

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == testingLabels.flatten()) * 100

        print("Evaluation completed with accuracy: {:.2f}%".format(accuracy))

    # Predict of data
    def predict(self, input):
        return self.feedForward(input)

    # Backpropagating all layers for the training
    def backpropagation(self, activations, preActivations, expectedOutput):
        outputError = activations[-1] - expectedOutput
        outputDelta = outputError * self.sigmoidDerivative(preActivations[-1])
        deltas = [outputDelta]

        for layerIndex in reversed(range(len(self.weights) - 1)):
            activationType = self.activationFunctions[layerIndex]
            derivative = self.derivativeFunction(preActivations[layerIndex], activationType)
            delta = deltas[0] @ self.weights[layerIndex+1].T * derivative
            deltas.insert(0, delta)

        for layerIndex in range(len(self.weights)):
            self.weights[layerIndex] -= self.learningRate * activations[layerIndex].T @ deltas[layerIndex]
            self.biases[layerIndex] -= self.learningRate * np.sum(deltas[layerIndex], axis=0, keepdims=True)

    # Feed forward for all the layers for one input
    def feedForward(self, inputData):
        activations = [inputData]
        preActivations = []
        currentActivation = inputData
        layersCount = len(self.weights)

        for layerIndex in range(layersCount):
            preActivation = np.dot(currentActivation, self.weights[layerIndex]) + self.biases[layerIndex]
            preActivations.append(preActivation)

            activationType = self.activationFunctions[layerIndex]
            currentActivation = self.activationFunction(preActivation, activationType)

            activations.append(currentActivation)

        return activations, preActivations

    # Deside function activation based on the activation type of each layer
    def activationFunction(self, data, activationType):
        if activationType == tools.RELU:
            return self.reluActivation(data)
        elif activationType == tools.SIGMOID:
            return self.sigmoidActivation(data)
        elif activationType == tools.TANH:
            return self.tanhActivation(data)
        else:
            return self.tanhActivation(data)

    # Deside function derivative based on the activation type of each layer
    def derivativeFunction(self, data, activationType):
        if activationType == tools.RELU:
            return self.reluDerivative(data)
        elif activationType == tools.SIGMOID:
            return self.sigmoidDerivative(data)
        elif activationType == tools.TANH:
            return self.tanhDerivative(data)
        else:
            return self.tanhDerivative(data)

    # Sigmoid function activation
    def sigmoidActivation(self, data):
        return 1 / (1 + np.exp(-data))

    # Sigmoid function derivative
    def sigmoidDerivative(self, data):
        s = self.sigmoidActivation(data)
        return s * (1 - s)

    # ReLU function activation
    def reluActivation(self, data):
        return np.maximum(0, data)

    # ReLU function derivative
    def reluDerivative(self, data):
        return (data > 0).astype(float)

    # Tanh function activation
    def tanhActivation(self, data):
        return np.tanh(data)

    # Tanh function derivative
    def tanhDerivative(self, data):
        return 1 - np.tanh(data) ** 2

    # Saving the trained model
    def saveModel(self, filename="NeuralNetworkModel.json"):
        print("\nSaving the model of neural network.")

        modelData = {
            "inputSize": self.inputLayerNeurons,
            "hiddenSize": self.hiddenLayersNeurons,
            "outputSize": self.outputLayerNeurons,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activationFunctions": self.activationFunctions
        }
        with open(filename, "w") as file:
            json.dump(modelData, file)

        print(f"Model saved in \"{filename}\"")

    # Load an already trained model
    def loadModel(self, modelFile="NeuralNetworkModel.json"):
        print(f"\nLoading model \"{modelFile}\"")

        with open(modelFile, "r") as file:
            modelData = json.load(file)

        self.inputLayerNeurons = modelData["inputSize"]
        self.hiddenLayersNeurons = modelData["hiddenSize"]
        self.outputLayerNeurons = modelData["outputSize"]
        self.weights = [np.array(w) for w in modelData["weights"]]
        self.biases = [np.array(b) for b in modelData["biases"]]
        self.activationFunctions = modelData.get("activationFunctions", None)

        print("Model loaded.")
