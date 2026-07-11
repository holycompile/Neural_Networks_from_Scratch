#this is the Strategy 2 which DOES WORK FOR THE VERTICAL DATA SET 
#Strategy 2: RANDOMLY ADJUST WEIGHTS AND BIASES- DOES WORK ( FOR VERTICAL DATA SET ONLY )
import numpy as np
import nnfs 
from nnfs.datasets import vertical_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights= 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases= np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights )+self.biases

#ReLU Activation
class Activation_ReLU:
    def forward(self,input):
        self.output=np.maximum(0,input)

#Softmax Actibvtion 
class Activation_Softmax:
    def forward(self, input):
        exp_values=np.exp(input - np.max(input, axis =1, keepdims=True))
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)
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
    

# Create dataset
X, y = vertical_data(samples=100, classes=3)
# Create model
dense1 = Layer_Dense(2, 3) # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()
# Create loss function
loss_function = Loss_CategoricalCrossentropy()


# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for iteration in range(10000):
 # Update weights with some small random values
 dense1.weights += 0.05 * np.random.randn(2, 3)
 dense1.biases += 0.05 * np.random.randn(1, 3)
 dense2.weights += 0.05 * np.random.randn(3, 3)
 dense2.biases += 0.05 * np.random.randn(1, 3)
 # Perform a forward pass of our training data through this layer
 dense1.forward(X)
 activation1.forward(dense1.output)
 dense2.forward(activation1.output)
 activation2.forward(dense2.output)
 # Perform a forward pass through activation function
 # it takes the output of second dense layer here and returns loss
 loss = loss_function.calculate(activation2.output, y)
 # Calculate accuracy from output of activation2 and targets
 # calculate values along first axis
 predictions = np.argmax(activation2.output, axis=1)
 accuracy = np.mean(predictions == y)
 # If loss is smaller - print and save weights and biases aside
 if loss < lowest_loss:
  print('New set of weights found, iteration:', iteration,'loss:', loss, 'acc:', accuracy)
  best_dense1_weights = dense1.weights.copy()
  best_dense1_biases = dense1.biases.copy()
  best_dense2_weights = dense2.weights.copy()
  best_dense2_biases = dense2.biases.copy()
  lowest_loss = loss
 # Revert weights and biases
 else:
  dense1.weights = best_dense1_weights.copy()
  dense1.biases = best_dense1_biases.copy()
  dense2.weights = best_dense2_weights.copy()
  dense2.biases = best_dense2_biases.copy()