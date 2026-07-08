import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Initialising the weights and biases 
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros( (1, n_neurons) )
    
    #Forward Pass: for doing the dot product
    def forward(self, inputs):
        #Calculating output values from inuts, weights and biases
        self.output = np.dot(inputs, self.weights)+self.biases

# Relu activation 
class ReluActivation:
    def forward(self,input):
        self.output=np.maximum(0,input)
        
X,y= spiral_data(samples=100, classes=3)

'''
dense1=Layer_Dense(2,3)
activation1=ReluActivation()
dense1.forward(X)
activation1.forward(dense1.output)
print(activation1.output[:5])
'''

# In my style 
dense1=Layer_Dense(2,3)
dense1.forward(X)
non_active_output=dense1.output

activation1=ReluActivation()
activation1.forward(non_active_output)
activated_output=activation1.output
print(activated_output[:5])