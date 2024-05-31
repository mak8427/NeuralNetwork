# Neural Network Training with Spiral Dataset

This project demonstrates the implementation and training of a simple neural network using the `nnfs` library. The neural network is designed to classify data from a spiral dataset into three classes. The project includes the implementation of various layers, activation functions, loss functions, and optimizers. Additionally, the training process is logged and visualized using Pandas and Matplotlib.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Classes and Functions](#classes-and-functions)
- [Training the Neural Network](#training-the-neural-network)
- [Visualizing Results](#visualizing-results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Ensure you have the necessary libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `nnfs`

You can install the required packages using pip:

```sh
pip install numpy pandas matplotlib nnfs
```

### Cloning the Repository

```sh
git clone https://github.com/your-username/nn-spiral-classification.git
cd nn-spiral-classification
```

## Usage

To train the neural network and visualize the results, run the following command:

```sh
python train_nn.py
```

## Classes and Functions

### `Layer_Dense`

This class represents a fully connected layer in the neural network.

- `__init__(self, n_inputs, n_neurons)`: Initializes the layer with random weights and zero biases.
- `forward(self, inputs)`: Performs a forward pass through the layer.
- `backward(self, dvalues)`: Performs a backward pass through the layer to calculate gradients.

### `Activation_ReLU`

This class represents the ReLU activation function.

- `forward(self, inputs)`: Applies the ReLU function to the inputs.
- `backward(self, dvalues)`: Calculates the gradient of the ReLU function.

### `Activation_Softmax`

This class represents the Softmax activation function.

- `forward(self, inputs)`: Applies the Softmax function to the inputs.
- `backward(self, dvalues)`: Calculates the gradient of the Softmax function.

### `Loss`

This is a base class for various loss functions.

- `calculate(self, output, y)`: Calculates the mean loss given the model output and ground truth values.

### `Loss_CategoricalCrossentropy`

This class represents the categorical cross-entropy loss function.

- `forward(self, y_pred, y_true)`: Calculates the forward pass for the loss.
- `backward(self, dvalues, y_true)`: Calculates the backward pass for the loss.

### `Activation_Softmax_Loss_CategoricalCrossentropy`

This class combines the Softmax activation function and the categorical cross-entropy loss for faster backward computation.

- `__init__(self)`: Initializes the Softmax activation and the categorical cross-entropy loss.
- `forward(self, inputs, y_true)`: Performs a forward pass and calculates the loss.
- `backward(self, dvalues, y_true)`: Performs a backward pass and calculates the gradient.

### Optimizers

Various optimizer classes are implemented to update the model parameters:

- `Optimizer_SGD`
- `optimizer_adagrad`
- `optimizer_RMS`
- `ADAM`

## Training the Neural Network

The training process involves the following steps:

1. **Data Preparation**: Load and preprocess the spiral dataset using `nnfs.datasets.spiral_data`.
2. **Model Initialization**: Initialize the dense layers, activation functions, and loss function.
3. **Training Loop**: Perform forward and backward passes, update model parameters using the optimizer, and log the training progress.
4. **Save Results**: Save the training data to a CSV file for later visualization.

## Visualizing Results

After training the model, you can visualize the training accuracy and loss over epochs using Matplotlib:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df1 = pd.read_csv('data_decay.csv')

plt.plot(df['accuracy'].head(10000), label='Without Decay')
plt.plot(df1['accuracy'], label='With Decay')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()
```

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. Ensure your code adheres to the existing style and includes comments where necessary.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
