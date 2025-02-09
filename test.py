import numpy as np
from sklearn.model_selection import train_test_split
import tools

def main():
    csvFile = "iris.csv"
    inputs, labels = tools.loadDataFromTextFile(csvFile, numberOfData=-1, inputSize=4, labelSize=1)
    labels = np.ones_like(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=42)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    
    # Create the neural network.
    # Configuration:
    #  - 4 input neurons (one for each feature)
    #  - Two hidden layers: first with 10 neurons and second with 5 neurons
    #  - 1 output neuron
    #  - Activation functions: ReLU for hidden layers and Sigmoid for output
    nn = tools.NeuralNetwork(
        inputLayerNeurons=4,
        hiddenLayersNeurons=[10, 5],
        outputLayerNeurons=1,
        activationFunctions=[tools.RELU, tools.RELU, tools.SIGMOID]
    )
    
    # Set training parameters.
    nn.setEpochs(1000)
    nn.setLearningRate(0.001)
    nn.setBatchSize(16)
    nn.setThreshold(1e-6)
    
    # Train the network.
    print("\nTraining the neural network...")
    nn.train(X_train, y_train)
    
    # Evaluate the network on the test set.
    nn.evaluation(X_test, y_test)
    
    # Interactive prediction.
    print("\nNow you can enter flower measurements to get a predicted probability.")
    print("Enter 4 comma-separated values corresponding to:")
    print("  Sepal Length, Sepal Width, Petal Length, Petal Width")
    print("For example: 6.0, 2.7, 5.1, 1.6")
    print("Type 'exit' or 'quit' to end.\n")
    
    userInput = input("Enter measurements (or 'exit' to quit): ")
    while userInput not in ["exit", "quit"]:
        try:
            features = [float(x.strip()) for x in userInput.split(",")]

            if len(features) != 4:
                print("Please enter exactly 4 values.\n")
                continue

            sample = np.array(features).reshape(1, -1)
            activations, _ = nn.feedForward(sample)
            probability = activations[-1][0, 0]

            print("Predicted probability that the flower is an iris: {:.2f}%".format(probability * 100))
            print()
        except Exception as e:
            print("Error processing input:", e, "\n")

        userInput = input("Enter measurements (or 'exit' to quit): ")

if __name__ == '__main__':
    main()
