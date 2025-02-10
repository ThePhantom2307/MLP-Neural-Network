import numpy as np
import json

# Activation function identifiers
RELU = "relu"
TANH = "tanh"
SIGMOID = "sigmoid"


class NeuralNetwork:
    """
    A neural network class for training and evaluating a multilayer perceptron.

    This class allows you to create a new neural network from scratch or load an existing model.
    It supports training via backpropagation, evaluating performance on test data, and saving/loading the model.

    Example:
        >>> nn = NeuralNetwork(input_layer_neurons=2, hidden_layers_neurons=[3], output_layer_neurons=1,
        ...                    activation_functions=[RELU, SIGMOID])
    """

    def __init__(self, input_layer_neurons=None, hidden_layers_neurons=None, output_layer_neurons=None,
                 model_file=None, activation_functions=None):
        """
        Initialize a NeuralNetwork instance.

        Either a new model is created (when no model_file is provided) or an existing model is loaded from a file.
        """
        self.input_layer_neurons = input_layer_neurons
        self.hidden_layers_neurons = hidden_layers_neurons or []
        self.output_layer_neurons = output_layer_neurons
        self.activation_functions = activation_functions

        self.check_initialization(input_layer_neurons, hidden_layers_neurons, output_layer_neurons,
                                  model_file, activation_functions)

        # Training parameters defaults
        self.epochs = 500
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.threshold = 1e-6

        self.weights = []
        self.biases = []

        if model_file:
            self.load_model(model_file)
        else:
            layers_neurons_size = self.hidden_layers_neurons + [self.output_layer_neurons]
            self.setup_weights_and_biases(layers_neurons_size)

    def check_initialization(self, input_layer_neurons, hidden_layers_neurons, output_layer_neurons,
                             model_file, activation_functions):
        """
        Validate initialization parameters.

        For a new model, all parameters must be provided and the number of activation functions must
        equal the number of hidden layers plus one (for the output layer).
        When loading a model (i.e. model_file is provided), the check is skipped.
        """
        # If loading a model, bypass additional configuration checks.
        if model_file is not None:
            return

        if input_layer_neurons is None or hidden_layers_neurons is None or \
           output_layer_neurons is None or activation_functions is None:
            raise ValueError(
                "Initialization Error: For a new model, you must provide input_layer_neurons, "
                "hidden_layers_neurons, output_layer_neurons, and activation_functions."
            )

        expected_activations = len(hidden_layers_neurons) + 1
        if len(activation_functions) != expected_activations:
            raise ValueError(
                f"Initialization Error: Expected {expected_activations} activation function(s) "
                f"(one for each hidden layer and one for the output layer), but received {len(activation_functions)}."
            )

    def correct_parameters_checking(self, input_data, labels):
        """
        Check that the dimensions of the input and label data match the network configuration.
        """
        if len(input_data[0]) != self.input_layer_neurons:
            raise ValueError(
                f"Parameter Error: Input size mismatch. Expected input size {self.input_layer_neurons} but received {len(input_data[0])}."
            )

        if len(labels[0]) != self.output_layer_neurons:
            raise ValueError(
                f"Parameter Error: Output size mismatch. Expected output size {self.output_layer_neurons} but received {len(labels[0])}."
            )

    def setup_weights_and_biases(self, layers_neurons_size):
        """
        Initialize the weights and biases randomly for the neural network layers.
        """
        print("\nSetting up the weights and biases of the neural network.")
        inputs_number = self.input_layer_neurons

        for neurons_number in layers_neurons_size:
            weight_matrix = np.random.rand(inputs_number, neurons_number) - 0.5
            bias_matrix = np.random.rand(1, neurons_number) - 0.5

            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

            inputs_number = neurons_number

        print("Setup completed.")

    def set_epochs(self, epochs):
        """Set the number of epochs for training."""
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        """Set the batch size for training."""
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        """Set the learning rate for training."""
        self.learning_rate = learning_rate

    def set_threshold(self, threshold):
        """Set the convergence threshold for training."""
        self.threshold = threshold

    def train(self, training_data, training_labels):
        """
        Train the neural network using the provided training data and labels.
        """
        print("\nStart training the neural network.")
        self.correct_parameters_checking(training_data, training_labels)

        number_samples = training_data.shape[0]
        for epoch in range(self.epochs):
            indices = np.arange(number_samples)
            np.random.shuffle(indices)
            training_data_shuffled = training_data[indices]
            training_labels_shuffled = training_labels[indices]

            total_epoch_loss = 0.0
            number_batches = 0

            for batch_start in range(0, number_samples, self.batch_size):
                batch_end = batch_start + self.batch_size
                batch_data = training_data_shuffled[batch_start:batch_end]
                batch_labels = training_labels_shuffled[batch_start:batch_end]

                activations, pre_activations = self.feed_forward(batch_data)
                predictions = activations[-1]
                batch_loss = np.mean((predictions - batch_labels) ** 2)
                total_epoch_loss += batch_loss
                number_batches += 1

                self.back_propagation(activations, pre_activations, batch_labels)

            average_loss = total_epoch_loss / number_batches
            print(f"Epoch {epoch+1}/{self.epochs} - Error: {average_loss:.6f}")

        print("Training completed.")

    def evaluation(self, testing_data, testing_labels):
        """
        Evaluate the performance of the neural network on test data.
        """
        print("\nStart evaluating the neural network.")
        self.correct_parameters_checking(testing_data, testing_labels)

        predictions = []
        for i in range(testing_data.shape[0]):
            sample = testing_data[i].reshape(1, -1)
            activations, _ = self.feed_forward(sample)
            output = activations[-1]

            predicted_label = 1 if output[0, 0] > 0.5 else 0
            predictions.append(predicted_label)

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == testing_labels.flatten()) * 100

        print("Evaluation completed with accuracy: {:.2f}%".format(accuracy))

    def predict(self, input_data):
        """
        Make a prediction for the given input data.
        """
        return self.feed_forward(input_data)

    def back_propagation(self, activations, pre_activations, expected_output):
        """
        Perform the backpropagation algorithm to update weights and biases.
        """
        output_error = activations[-1] - expected_output
        output_delta = output_error * self.sigmoid_derivative(pre_activations[-1])
        deltas = [output_delta]

        for layer_index in reversed(range(len(self.weights) - 1)):
            activation_type = self.activation_functions[layer_index]
            derivative = self.derivative_function(pre_activations[layer_index], activation_type)
            delta = deltas[0] @ self.weights[layer_index + 1].T * derivative
            deltas.insert(0, delta)

        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= self.learning_rate * activations[layer_index].T @ deltas[layer_index]
            self.biases[layer_index] -= self.learning_rate * np.sum(deltas[layer_index], axis=0, keepdims=True)

    def feed_forward(self, input_data):
        """
        Compute the feed-forward pass for the network.
        """
        activations = [input_data]
        pre_activations = []
        current_activation = input_data
        layers_count = len(self.weights)

        for layer_index in range(layers_count):
            pre_activation = np.dot(current_activation, self.weights[layer_index]) + self.biases[layer_index]
            pre_activations.append(pre_activation)

            activation_type = self.activation_functions[layer_index]
            current_activation = self.activation_function(pre_activation, activation_type)
            activations.append(current_activation)

        return activations, pre_activations

    def activation_function(self, data, activation_type):
        """
        Apply the specified activation function to the data.
        """
        if activation_type == RELU:
            return self.relu_activation(data)
        elif activation_type == SIGMOID:
            return self.sigmoid_activation(data)
        elif activation_type == TANH:
            return self.tanh_activation(data)
        else:
            return self.tanh_activation(data)

    def derivative_function(self, data, activation_type):
        """
        Compute the derivative of the activation function.
        """
        if activation_type == RELU:
            return self.relu_derivative(data)
        elif activation_type == SIGMOID:
            return self.sigmoid_derivative(data)
        elif activation_type == TANH:
            return self.tanh_derivative(data)
        else:
            return self.tanh_derivative(data)

    def sigmoid_activation(self, data):
        """
        Apply the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-data))

    def sigmoid_derivative(self, data):
        """
        Compute the derivative of the sigmoid function.
        """
        s = self.sigmoid_activation(data)
        return s * (1 - s)

    def relu_activation(self, data):
        """
        Apply the ReLU activation function.
        """
        return np.maximum(0, data)

    def relu_derivative(self, data):
        """
        Compute the derivative of the ReLU function.
        """
        return (data > 0).astype(float)

    def tanh_activation(self, data):
        """
        Apply the tanh activation function.
        """
        return np.tanh(data)

    def tanh_derivative(self, data):
        """
        Compute the derivative of the tanh function.
        """
        return 1 - np.tanh(data) ** 2

    def save_model(self, filename="NeuralNetworkModel.json"):
        """
        Save the current model to a JSON file.
        """
        print("\nSaving the model of neural network.")
        model_data = {
            "input_size": self.input_layer_neurons,
            "hidden_size": self.hidden_layers_neurons,
            "output_size": self.output_layer_neurons,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activation_functions": self.activation_functions
        }
        with open(filename, "w") as file:
            json.dump(model_data, file)

        print(f"Model saved in \"{filename}\"")

    def load_model(self, model_file="NeuralNetworkModel.json"):
        """
        Load a model from a JSON file.
        """
        print(f"\nLoading model \"{model_file}\"")
        with open(model_file, "r") as file:
            model_data = json.load(file)

        self.input_layer_neurons = model_data["input_size"]
        self.hidden_layers_neurons = model_data["hidden_size"]
        self.output_layer_neurons = model_data["output_size"]
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]
        self.activation_functions = model_data.get("activation_functions")

        print("Model loaded.")


# Example main script using the NeuralNetwork class
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    # Dummy implementation for loading data from a CSV file.
    # Replace this with your actual implementation.
    def load_data_from_text_file(filename, number_of_data=-1, input_size=4, label_size=1):
        # For example purposes, generate random data:
        data = np.random.rand(150, input_size + label_size)
        inputs = data[:, :input_size]
        labels = data[:, input_size:]
        return inputs, labels

    csv_file = "iris.csv"
    inputs, labels = load_data_from_text_file(csv_file, number_of_data=-1, input_size=4, label_size=1)
    # Overwrite labels to simulate a binary classification problem
    labels = np.ones_like(labels)

    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=42)
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    # Create a new neural network model
    neural_network = NeuralNetwork(
        input_layer_neurons=4,
        hidden_layers_neurons=[10, 5],
        output_layer_neurons=1,
        activation_functions=[RELU, RELU, SIGMOID]
    )

    neural_network.set_epochs(1000)
    neural_network.set_learning_rate(0.001)
    neural_network.set_batch_size(16)
    neural_network.set_threshold(1e-6)

    print("\nTraining the neural network...")
    neural_network.train(x_train, y_train)

    neural_network.evaluation(x_test, y_test)

    neural_network.save_model("test.json")

    # Load the saved model (note that you no longer need to pass activation_functions)
    neural_network = NeuralNetwork(model_file="test.json")

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
