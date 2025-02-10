import numpy as np
import pandas as pd
from random import randint

def load_data_from_text_file(file_path, number_of_data=-1, input_size=2, label_size=1):
    """
    Load a dataset from a CSV text file and split it into input features and labels.

    This function reads a CSV file (without headers) from the specified file path using pandas.
    It can optionally limit the number of rows read from the file. The data is then split into two
    parts: one for input features and one for labels, based on the provided column sizes.

    Args:
        file_path (str): The path to the CSV file containing the dataset.
        number_of_data (int, optional): The maximum number of data rows to load. If set to -1 (default),
            all rows are loaded.
        input_size (int, optional): The number of columns representing input features. Defaults to 2.
        label_size (int, optional): The number of columns representing labels. Defaults to 1.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - np.ndarray: The array of input features.
            - np.ndarray: The array of labels.

    Example:
        >>> inputs, labels = load_data_from_text_file("data.csv", number_of_data=100, input_size=3, label_size=1)
        >>> print(inputs.shape, labels.shape)
    """
    print(f"\nLoading dataset from file \"{file_path}\"")
    data = pd.read_csv(file_path, header=None)

    if number_of_data != -1:
        data = data.iloc[:number_of_data]

    data = data.values

    input_data = data[:, :input_size]
    labels = data[:, input_size:input_size + label_size]
    
    print("Dataset loaded.")
    return np.array(input_data), np.array(labels)


def generate_example_dataset(number_of_data):
    """
    Generate an example XOR dataset.

    This function creates a dataset for the XOR logic gate by generating a specified number of
    random binary pairs. For each pair, the XOR (exclusive OR) operation is computed to produce a label.
    The dataset is returned as two lists: one containing the input pairs and the other containing the
    corresponding XOR results.

    Args:
        number_of_data (int): The number of data samples to generate.

    Returns:
        tuple: A tuple containing two elements:
            - input_data (list of list of int): A list where each element is a list of two binary integers,
              representing the input pair (e.g., [0, 1]).
            - labels (list of int): A list of integers (0 or 1) representing the XOR result for each input pair.

    Example:
        >>> inputs, labels = generate_example_dataset(4)
        >>> print(inputs)
        [[0, 1], [1, 0], [0, 0], [1, 1]]
        >>> print(labels)
        [1, 1, 0, 0]
    """
    input_data = []
    labels = []

    for _ in range(number_of_data):
        x_random = randint(0, 1)
        y_random = randint(0, 1)

        random_data = [x_random, y_random]
        result = x_random ^ y_random
        
        input_data.append(random_data)
        labels.append(result)

    return input_data, labels
