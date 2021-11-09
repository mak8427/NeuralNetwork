import nnfs
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import math
import time

nnfs.init()
np.random.seed(0)


E=math.e

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def foward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases
class Activation_ReLU:
    def foward(self,inputs):
        self.output=np.maximum(0,inputs)
class Soft_Max:
    def foward(self,inputs):
        exp=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        prob=exp/np.sum(exp,axis=1,keepdims=True)
        self.output=prob
class Loss:
    def calculate(self,output,y):
        sample_losses=self.foward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
class Loss_CatCrosEnt(Loss):
    def foward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7, 1-(1e-7))
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)
        negative_log=np.log(correct_confidences)
        return negative_log

X,y= spiral_data(samples= 100,classes=3)

dense1=Layer_Dense(2,3)
activation1=Activation_ReLU()

dense2=Layer_Dense(3,3)
activation2=Soft_Max()

dense1.foward(X)
activation1.foward(dense1.output)

dense2.foward(activation1.output)
activation2.foward(dense2.output)


loss_fun=Loss_CatCrosEnt()
loss=Loss.calculate(activation2.output, y)

print(loss)