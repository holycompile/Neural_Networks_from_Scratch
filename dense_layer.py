import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

#Dense Layers

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Initialising the weights and biases 
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros( (1, n_neurons) )
    
    #Forward Pass: for doing the dot product
    def forward(self, inputs):
        #Calculating output values from inuts, weights and biases
        self.output = np.dot(inputs, self.weights)+self.biases
        
#Dataset create
X,y =spiral_data(samples=100, classes=3)
    
dense1 = Layer_Dense(2,3) # 2 inputs and 3 neurons 
dense1.forward(X)
    
print(dense1.output)