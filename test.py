import numpy as np
import NeuralNetworkMLP as nn

def main():
    # Loading a dataset from the csv file "iris.csv" which have sets of 4 numbers as inputs and 1 number as output
    csvFile = "iris.csv"
    inputs, labels = nn.loadDataFromTextFile(csvFile, numberOfData=-1, inputSize=4, labelSize=1)
    labels = np.ones_like(labels)
    
    # split the dataset: 30% for testing, 70% for training.
    n = inputs.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.3 * n)  # 30% for test data
    test_indices = indices[:split]
    train_indices = indices[split:]
    
    xTest = inputs[test_indices]
    yTest = labels[test_indices]
    xTrain = inputs[train_indices]
    yTrain = labels[train_indices]
    
    print("Training data shape:", xTrain.shape)
    print("Testing data shape:", xTest.shape)
    
    # Initializing the neural network structure
    # Input layer neurons: 4
    # First hidden layer neurons: 10 (ReLU activation)
    # Second hidden layer neurons: 5 (ReLU activation)
    # Output layer neurons: 1 (Sigmoid activation)
    neuralNetwork = nn.NeuralNetwork(
        inputLayerNeurons=4,
        hiddenLayersNeurons=[10, 5],
        outputLayerNeurons=1,
        activationFunctions=[nn.RELU, nn.RELU, nn.SIGMOID]
    )

    # Setting up the parameters of the neural network
    # Epochs: 1000
    # Batch size: 16
    # Learning rate: 0.001
    # Threshold: 1e-6
    neuralNetwork.setEpochs(1000)
    neuralNetwork.setBatchSize(16)
    neuralNetwork.setLearningRate(0.001)
    neuralNetwork.setThreshold(1e-6)
    
    # Training the neural network with the xTrain and yTrain datasets
    print("\nTraining the neural network...")
    neuralNetwork.train(xTrain, yTrain)
    
    # Check the accuracy of the trained model using the xTest and yTest datasets
    neuralNetwork.evaluation(xTest, yTest)

    # Saving the trained model in the file "test.json"
    neuralNetwork.saveModel("test.json")

    # Clear the current network instance (Not required, it is just for testing).
    neuralNetwork = None

    # Load the saved model from file "test.json".
    neuralNetwork = nn.NeuralNetwork(modelFile="test.json")
    
    # A loop where user can input values and the neural network will predict the output of them
    print("\nNow you can enter flower measurements to get a predicted probability.")
    print("Enter 4 comma-separated values corresponding to:")
    print("  Sepal Length, Sepal Width, Petal Length, Petal Width")
    print("For example: 6.0, 2.7, 5.1, 1.6")
    print("Type 'exit' or 'quit' to end.\n")
    
    userInput = input("Enter measurements (or 'exit' to quit): ")
    while userInput.lower() not in ["exit", "quit"]:
        try:
            features = [float(x.strip()) for x in userInput.split(",")]
            if len(features) != 4:
                print("Please enter exactly 4 values.\n")
                userInput = input("Enter measurements (or 'exit' to quit): ")
                continue

            sample = np.array(features).reshape(1, -1)
            activations, _ = neuralNetwork.feedForward(sample)
            probability = activations[-1][0, 0]

            print("Predicted probability that the flower is an iris: {:.2f}%".format(probability * 100))
            print()
        except Exception as e:
            print("Error processing input:", e, "\n")

        userInput = input("Enter measurements (or 'exit' to quit): ")

if __name__ == '__main__':
    main()
