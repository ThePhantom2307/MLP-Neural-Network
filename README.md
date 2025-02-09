MLP Neural Network
MLP Neural Network is a lightweight Python library for building and training Multi-Layer Perceptron (MLP) models. This repository provides a simple yet powerful framework for experimenting with neural networks, along with an example program and dataset to help you get started quickly.

Features
Simple API: Quickly build and train neural network models with a few lines of code.
Customizable Architecture: Easily configure the number of layers, neurons per layer, activation functions, and learning rates.
Example Program: The included test.py script demonstrates how to integrate and use the library in a real-world scenario.
Sample Dataset: Comes with an iris.csv file containing the famous Iris dataset, ideal for testing and learning.
Getting Started
Prerequisites
Python 3.x is required.

Ensure you have the necessary libraries installed. You might need packages such as NumPy and Pandas. If you have a list of dependencies, consider creating a requirements.txt file. For example:

bash
Copy
pip install numpy pandas
Installation
Clone the repository to your local machine:

bash
Copy
git clone https://github.com/ThePhantom2307/MLP-Neural-Network.git
cd MLP-Neural-Network
Running the Example
The repository includes an example script (test.py) that shows how to use the library. The script loads the provided iris.csv dataset, builds an MLP model, trains it, and evaluates its performance. To run the example, simply execute:

bash
Copy
python test.py
Usage
You can incorporate the MLP Neural Network library into your own projects. Below is a short example to illustrate how to use the library:

python
Copy
# Import the MLP model from the library (adjust the import as needed)
from mlp_neural_network import MLP

# Define your network architecture: for example, input layer of size 4, one hidden layer with 10 neurons, and output layer of size 3.
model = MLP(layers=[4, 10, 3], activation='sigmoid', learning_rate=0.01)

# Train the model on your training data (X_train and y_train should be defined appropriately)
model.train(X_train, y_train, epochs=100)

# Predict on new data
predictions = model.predict(X_test)
For more detailed usage, refer to the test.py file, which provides a complete example.

Contributing
Contributions, suggestions, and bug reports are welcome! If you’d like to contribute, please fork the repository and create a pull request. Alternatively, feel free to open an issue to discuss improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.
