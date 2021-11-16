import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pandas as pd
import time

nnfs.init()


class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
      # Gradients on parameters
      self.dweights = np.dot(self.inputs.T, dvalues)
      self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
      # Gradient on values
      self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs=inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<=0]=0


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backwards(self,dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(- 1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)



# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy ():
    # Creates activation and loss function objects
    def __init__ ( self ):
       self.activation = Activation_Softmax()
       self.loss = Loss_CategoricalCrossentropy()
    # Forward pass
    def forward ( self , inputs , y_true ):
    # Output layer's activation function
       self.activation.forward(inputs)
    # Set the output
       self.output = self.activation.output
    # Calculate and return loss value
       return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
       # Number of samples
       samples = len(dvalues)
       # If labels are one-hot encoded,
       # turn them into discrete values
       if len(y_true.shape) == 2:
           y_true = np.argmax(y_true, axis=1)
       # Copy so we can safely modify
       self.dinputs = dvalues.copy()
       # Calculate gradient
       self.dinputs[range(samples), y_true] -= 1
       # Normalize gradient
       self.dinputs = self.dinputs / samples
#Optimizer
class Optimizer_SGD:
    def __init__(self,learning_rate=1.0,decay=0.,momentum=0):
        self.learning_rate = learning_rate
        self.current_lerning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum
    def pre_update_paramns(self):
        if self.decay:
           self.current_lerning_rate=self.learning_rate/(1.+self.decay*self.iterations)
    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums=np.zeros_like(layer.weights)
                layer.bias_momentums=np.zeros_like(layer.biases)
            weight_updates=layer.weight_momentums*self.momentum-self.current_lerning_rate*layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates=layer.bias_momentums*self.momentum-self.current_lerning_rate*layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            bias_updates = -self.current_lerning_rate* layer.dbiases
            weight_updates = -self.current_lerning_rate* layer.dweights
        layer.weights += weight_updates
        layer.biases += bias_updates


    def post_update_paramns(self):
        self.iterations+=1


# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 64 )

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense( 64 , 3 )

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#oprimizer
optimizer = Optimizer_SGD( decay = 1e-3 ,momentum=0.9)

data=pd.DataFrame(columns=['epoch','accuracy','loss'])
for epoch in range(0,10000):
     start_time = time.time()
    # Perform a forward pass of our training data through this layer
     dense1.forward(X)

     # Perform a forward pass through activation function
     # takes the output of first dense layer here
     activation1.forward(dense1.output)

     # Perform a forward pass through second Dense layer
     # takes outputs of activation function of first layer as inputs
     dense2.forward(activation1.output)

     # Perform a forward pass through the activation/loss function
     # takes the output of second dense layer here and returns loss
     loss = loss_activation.forward(dense2.output, y)

     # Calculate accuracy from output of activation2 and targets
     # calculate values along first axis
     predictions = np.argmax(loss_activation.output, axis = 1 )
     if len (y.shape) == 2 :
        y = np.argmax(y, axis = 1 )
     accuracy = np.mean(predictions == y)

     # Print accuracy
     if not epoch % 100:
         print(str(epoch)+'  '+ str(accuracy)+ '  '+ str(loss)+'  '+str(optimizer.current_lerning_rate))
     data.loc[len(data)]=[epoch,accuracy,loss]


     # Backward pass
     loss_activation.backward(loss_activation.output, y)
     dense2.backward(loss_activation.dinputs)
     activation1.backward(dense2.dinputs)
     dense1.backward(activation1.dinputs)

     optimizer.pre_update_paramns()
     optimizer.update_params(dense1)
     optimizer.update_params(dense2)
     optimizer.post_update_paramns()
     end_time = time.time()
     total_time = end_time - start_time
     if not epoch % 100:
         print(total_time)

data.to_csv('data_decay.csv')