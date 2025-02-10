import numpy as np
from sklearn.model_selection import train_test_split
import NeuralNetworkMLP as nn

def main():
    csv_file = "iris.csv"
    inputs, labels = nn.load_data_from_text_file(csv_file, number_of_data=-1, input_size=4, label_size=1)
    labels = np.ones_like(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=42)
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
    
    neural_network = nn.NeuralNetwork(
        input_layer_neurons=4,
        hidden_layers_neurons=[10, 5],
        output_layer_neurons=1,
        activation_functions=[nn.RELU, nn.RELU, nn.SIGMOID]
    )
    
    neural_network.set_epochs(1000)
    neural_network.set_learning_rate(0.001)
    neural_network.set_batch_size(16)
    neural_network.set_threshold(1e-6)
    
    print("\nTraining the neural network...")
    neural_network.train(x_train, y_train)
    
    neural_network.evaluation(x_test, y_test)

    neural_network.save_model("test.json")

    neural_network = None

    neural_network = nn.NeuralNetwork(model_file="test.json")
    
    print("\nNow you can enter flower measurements to get a predicted probability.")
    print("Enter 4 comma-separated values corresponding to:")
    print("  Sepal Length, Sepal Width, Petal Length, Petal Width")
    print("For example: 6.0, 2.7, 5.1, 1.6")
    print("Type 'exit' or 'quit' to end.\n")
    
    user_input = input("Enter measurements (or 'exit' to quit): ")
    while user_input.lower() not in ["exit", "quit"]:
        try:
            features = [float(x.strip()) for x in user_input.split(",")]
            if len(features) != 4:
                print("Please enter exactly 4 values.\n")
                user_input = input("Enter measurements (or 'exit' to quit): ")
                continue

            sample = np.array(features).reshape(1, -1)
            activations, _ = neural_network.feed_forward(sample)
            probability = activations[-1][0, 0]

            print("Predicted probability that the flower is an iris: {:.2f}%".format(probability * 100))
            print()
        except Exception as e:
            print("Error processing input:", e, "\n")

        user_input = input("Enter measurements (or 'exit' to quit): ")

if __name__ == '__main__':
    main()
