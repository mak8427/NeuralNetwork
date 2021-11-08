import nnfs
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
nnfs.init()
np.random.seed(0)

X,y= spiral_data(100,3)


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def foward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases
class Activation_ReLU:
    def foward(self,inputs):
        self.output=np.maximum(0,inputs)
layer1=Layer_Dense(2,5)
layer1.foward(X)
activation1=Activation_ReLU()
layer1.foward(X)
activation1.foward(layer1.output)
print(activation1.output)


inputs=[0,2,-1,3.3,-2.7,1.1,2.2,100]

