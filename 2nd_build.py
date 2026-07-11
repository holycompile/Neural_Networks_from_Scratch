# 1st Attempt for forward implementation without Loss 
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
    
#Softmax_activation   
class Activation_Softmax:
    def forward(self,input):
        exp_values=np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities
 
 
 
#common Loss class
class Loss:
    def calculate(self, output, y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
          
          
                 
#Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        
        samples=len(y_pred)
        #Clipping the data to prevent division by zero
        y_pred_clipped=np.clip(y_pred, 1e-7, 1- 1e-7)
        
        #Case 1: Non-hot-encoded , recall !!, see entropy_loss.py
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples), y_true]
        #Clase 2: Hot-encoded, see entropy_2.0.py
        elif len(y_true.shape)==2:
            correct_confidences= np.sum(y_pred_clipped*y_true, axis=1)
        #Losses
        negetive_log_likelihoods= -np.log(correct_confidences)
        return negetive_log_likelihoods
    

    
    
X,y= spiral_data(samples=100, classes=3)
#------------------------------------------------------
dense1=Layer_Dense(2,3)
activation1=ReluActivation()

dense2=Layer_Dense(3,3)
activation2=Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
#-------------------------------------------------------

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print("loss:", loss)

predictions= np.argmax(activation2.output, axis=1)
if len(y.shape)==2:
    y=np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print("Accuracy", accuracy)