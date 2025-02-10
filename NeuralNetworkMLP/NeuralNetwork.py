import numpy as np
import json

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

        This constructor sets up the neural network either for training a new model (using the provided
        layer configuration and activation functions) or for loading an existing model from a file.

        Args:
            input_layer_neurons (int, optional): Number of neurons in the input layer.
            hidden_layers_neurons (list of int, optional): A list where each element is the number of neurons in a hidden layer.
            output_layer_neurons (int, optional): Number of neurons in the output layer.
            model_file (str, optional): Path to a JSON file containing a saved model. If provided, the model is loaded from this file.
            activation_functions (list, optional): A list of activation functions for each layer.
                The list should have a length equal to the number of hidden layers plus one (for the output layer).

        Raises:
            ValueError: If all parameters are None, if both a model file and full layer configuration are provided,
                        or if there is a mismatch in the number of activation functions.

        Example:
            >>> nn = NeuralNetwork(input_layer_neurons=2, hidden_layers_neurons=[3], output_layer_neurons=1,
            ...                    activation_functions=[RELU, SIGMOID])
        """
        self.input_layer_neurons = input_layer_neurons
        self.hidden_layers_neurons = hidden_layers_neurons or []
        self.output_layer_neurons = output_layer_neurons
        self.activation_functions = activation_functions

        self.epochs = 500
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.threshold = 1e-6

        self.weights = []
        self.biases = []

        if model_file:
            self.load_model(model_file)
            self.check_initialization(input_layer_neurons, hidden_layers_neurons, output_layer_neurons, model_file, activation_functions)
        else:
            layers_neurons_size = self.hidden_layers_neurons + [self.output_layer_neurons]
            self.check_initialization(input_layer_neurons, hidden_layers_neurons, output_layer_neurons, model_file, activation_functions)
            self.setup_weights_and_biases(layers_neurons_size)

    def check_initialization(self, input_layer_neurons, hidden_layers_neurons, output_layer_neurons, model_file, activation_functions):
        """
        Validate initialization parameters.

        Ensures that either a model file is provided to load an existing model or that the full layer configuration
        is provided to build a new model—but not both. Also checks that the number of activation functions matches the
        number of layers that require them.

        Args:
            input_layer_neurons (int): Number of neurons in the input layer.
            hidden_layers_neurons (list): Neurons in the hidden layers.
            output_layer_neurons (int): Number of neurons in the output layer.
            model_file (str): Filename for an existing model.
            activation_functions (list): Activation functions for each layer.

        Raises:
            ValueError: If all parameters are None, if both types of parameters are provided,
                        or if there is a mismatch in the expected activation functions.

        Example:
            >>> # This will raise a ValueError if activation_functions length does not match hidden layers + output.
            >>> nn = NeuralNetwork(2, [3, 4], 1, activation_functions=[RELU, SIGMOID])
        """

        if all(param is None for param in [input_layer_neurons, hidden_layers_neurons, output_layer_neurons, model_file, activation_functions]):
            raise ValueError(
                "Initialization Error: All parameters are None. You must provide either a model_file "
                "to load an existing model OR provide the layer configuration to build a new model."
            )

        if all(param is not None for param in [input_layer_neurons, hidden_layers_neurons, output_layer_neurons, model_file, activation_functions]):
            raise ValueError(
                "Initialization Error: Conflicting parameters provided. Provide either a model_file "
                "OR the explicit layer parameters, but not both."
            )

        expected_activations = len(self.hidden_layers_neurons) + 1
        if self.activation_functions is None or len(self.activation_functions) != expected_activations:
            received = 0 if activation_functions is None else len(activation_functions)
            raise ValueError(
                f"Initialization Error: Expected {expected_activations} activation function(s) (one for each hidden layer and one for the output layer), but received {received}."
            )

    def correct_parameters_checking(self, input_data, labels):
        """
        Check that the dimensions of the input and label data match the network configuration.

        Args:
            input_data (array-like): The input data (assumed to be a 2D array where each row is an example).
            labels (array-like): The label data corresponding to the input data.

        Raises:
            ValueError: If the size of the input or label data does not match the expected number of neurons.

        Example:
            >>> # Assuming the network expects 2 inputs and 1 output:
            >>> nn.correct_parameters_checking([[0.1, 0.2]], [[1]])
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

        This method sets up the weight matrices and bias vectors for each layer based on the network architecture.

        Args:
            layers_neurons_size (list of int): List containing the number of neurons for each layer (hidden layers and output layer).

        Example:
            >>> nn = NeuralNetwork(input_layer_neurons=2, hidden_layers_neurons=[3], output_layer_neurons=1,
            ...                    activation_functions=[RELU, SIGMOID])
            >>> nn.setup_weights_and_biases([3, 1])
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
        """
        Set the number of epochs for training.

        Args:
            epochs (int): The number of training epochs.

        Example:
            >>> nn.set_epochs(1000)
        """
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        """
        Set the batch size for training.

        Args:
            batch_size (int): The batch size.

        Example:
            >>> nn.set_batch_size(64)
        """
        self.batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        """
        Set the learning rate for training.

        Args:
            learning_rate (float): The learning rate.

        Example:
            >>> nn.set_learning_rate(0.001)
        """
        self.learning_rate = learning_rate

    def set_threshold(self, threshold):
        """
        Set the convergence threshold for training.

        Args:
            threshold (float): The threshold value.

        Example:
            >>> nn.set_threshold(1e-5)
        """
        self.threshold = threshold

    def train(self, training_data, training_labels):
        """
        Train the neural network using the provided training data and labels.

        This method uses backpropagation to adjust the weights and biases over multiple epochs.

        Args:
            training_data (np.array): The input data for training.
            training_labels (np.array): The expected output for each training example.

        Example:
            >>> import numpy as np
            >>> # Create dummy data: 100 samples, 2 features per sample, 1 label per sample.
            >>> training_data = np.random.rand(100, 2)
            >>> training_labels = np.random.rand(100, 1)
            >>> nn.train(training_data, training_labels)
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

        Args:
            testing_data (np.array): The input data for testing.
            testing_labels (np.array): The true labels for the test data.

        Prints:
            The accuracy of the model as a percentage.

        Example:
            >>> # Assuming testing_data and testing_labels have appropriate shapes:
            >>> nn.evaluation(testing_data, testing_labels)
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

        Args:
            input_data (np.array): Input data for which to predict the output.

        Returns:
            tuple: A tuple containing the activations and pre-activations from the feed-forward pass.

        Example:
            >>> # Assuming input_data has the correct dimensions:
            >>> activations, pre_activations = nn.predict(np.array([[0.5, 0.5]]))
        """
        return self.feed_forward(input_data)

    def back_propagation(self, activations, pre_activations, expected_output):
        """
        Perform the backpropagation algorithm to update weights and biases.

        Args:
            activations (list of np.array): Activations from each layer, including the input.
            pre_activations (list of np.array): Linear combinations (z values) before applying the activation function.
            expected_output (np.array): The expected output values.

        Updates:
            The network's weights and biases are adjusted in-place.

        Example:
            >>> # This is normally called internally by train().
            >>> nn.back_propagation(activations, pre_activations, expected_output)
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

        Args:
            input_data (np.array): The input data.

        Returns:
            tuple: A tuple (activations, pre_activations) where:
                - activations is a list of activations for each layer,
                - pre_activations is a list of the linear combinations (z values) before applying the activation function.

        Example:
            >>> activations, pre_activations = nn.feed_forward(np.array([[0.5, 0.5]]))
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

        Args:
            data (np.array): Input data.
            activation_type (str): The type of activation function (e.g., RELU, SIGMOID, TANH).

        Returns:
            np.array: The activated output.

        Example:
            >>> result = nn.activation_function(np.array([0.0]), RELU)
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

        Args:
            data (np.array): The pre-activation values.
            activation_type (str): The type of activation function used.

        Returns:
            np.array: The derivative of the activation function evaluated at the given data.

        Example:
            >>> derivative = nn.derivative_function(np.array([0.0]), SIGMOID)
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

        Args:
            data (np.array): Input data.

        Returns:
            np.array: The result of applying the sigmoid function.

        Example:
            >>> result = nn.sigmoid_activation(np.array([0.0]))
        """
        return 1 / (1 + np.exp(-data))

    def sigmoid_derivative(self, data):
        """
        Compute the derivative of the sigmoid function.

        Args:
            data (np.array): Pre-activation values.

        Returns:
            np.array: The derivative of the sigmoid function.

        Example:
            >>> deriv = nn.sigmoid_derivative(np.array([0.0]))
        """
        s = self.sigmoid_activation(data)
        return s * (1 - s)

    def relu_activation(self, data):
        """
        Apply the ReLU activation function.

        Args:
            data (np.array): Input data.

        Returns:
            np.array: The result of applying the ReLU function.

        Example:
            >>> result = nn.relu_activation(np.array([-1, 0, 1]))
        """
        return np.maximum(0, data)

    def relu_derivative(self, data):
        """
        Compute the derivative of the ReLU function.

        Args:
            data (np.array): Pre-activation values.

        Returns:
            np.array: The derivative of the ReLU function.

        Example:
            >>> deriv = nn.relu_derivative(np.array([-1, 0, 1]))
        """
        return (data > 0).astype(float)

    def tanh_activation(self, data):
        """
        Apply the tanh activation function.

        Args:
            data (np.array): Input data.

        Returns:
            np.array: The result of applying the tanh function.

        Example:
            >>> result = nn.tanh_activation(np.array([0.0]))
        """
        return np.tanh(data)

    def tanh_derivative(self, data):
        """
        Compute the derivative of the tanh function.

        Args:
            data (np.array): Pre-activation values.

        Returns:
            np.array: The derivative of the tanh function.

        Example:
            >>> deriv = nn.tanh_derivative(np.array([0.0]))
        """
        return 1 - np.tanh(data) ** 2

    def save_model(self, filename="NeuralNetworkModel.json"):
        """
        Save the current model to a JSON file.

        The model data, including the network architecture, weights, biases, and activation functions, is saved
        in JSON format.

        Args:
            filename (str): The file name to which the model will be saved.

        Example:
            >>> nn.save_model("mymodel.json")
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

        This method updates the network's parameters, weights, biases, and activation functions using the data
        from the provided JSON file.

        Args:
            model_file (str): The path to the JSON file containing the model data.

        Example:
            >>> nn.load_model("mymodel.json")
        """
        print(f"\nLoading model \"{model_file}\"")
        with open(model_file, "r") as file:
            model_data = json.load(file)

        self.input_layer_neurons = model_data["input_size"]
        self.hidden_layers_neurons = model_data["hidden_size"]
        self.output_layer_neurons = model_data["output_size"]
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]
        self.activation_functions = model_data.get("activation_functions", None)

        print("Model loaded.")
