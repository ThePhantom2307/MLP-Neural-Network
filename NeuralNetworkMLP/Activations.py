import numpy as np
    
def sigmoidActivation(data):
    """
    Apply the sigmoid activation function.

    Args:
        data (np.array): Input data.

    Returns:
        np.array: The result of applying the sigmoid function.

    Example:
        >>> result = nn.sigmoidActivation(np.array([0.0]))
    """
    return 1 / (1 + np.exp(-data))

def sigmoidDerivative(data):
    """
    Compute the derivative of the sigmoid function.

    Args:
        data (np.array): Pre-activation values.

    Returns:
        np.array: The derivative of the sigmoid function.

    Example:
        >>> deriv = nn.sigmoidDerivative(np.array([0.0]))
    """
    s = sigmoidActivation(data)
    return s * (1 - s)

def reluActivation(data):
    """
    Apply the ReLU activation function.

    Args:
        data (np.array): Input data.

    Returns:
        np.array: The result of applying the ReLU function.

    Example:
        >>> result = nn.reluActivation(np.array([-1, 0, 1]))
    """
    return np.maximum(0, data)

def reluDerivative(data):
    """
    Compute the derivative of the ReLU function.

    Args:
        data (np.array): Pre-activation values.

    Returns:
        np.array: The derivative of the ReLU function.

    Example:
        >>> deriv = nn.reluDerivative(np.array([-1, 0, 1]))
    """
    return (data > 0).astype(float)

def tanhActivation(data):
    """
    Apply the tanh activation function.

    Args:
        data (np.array): Input data.

    Returns:
        np.array: The result of applying the tanh function.

    Example:
        >>> result = nn.tanhActivation(np.array([0.0]))
    """
    return np.tanh(data)

def tanhDerivative(data):
    """
    Compute the derivative of the tanh function.

    Args:
        data (np.array): Pre-activation values.

    Returns:
        np.array: The derivative of the tanh function.

    Example:
        >>> deriv = nn.tanhDerivative(np.array([0.0]))
    """
    return 1 - np.tanh(data) ** 2