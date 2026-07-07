import numpy as np 


# single neuron 

input = [ 1.0, 2.0, 3.0, 2.5]
weight= [ 0.2, 0.8, -0.55, 1.0]
bias = 2.0

input_array=np.array(input) # or directly write input_array=np.array([ 1, 2, 3, 2.5])
weight_array=np.array(weight)

output=np.dot(input_array, weight_array) + bias 

print("----------------------------------")
print("Single Neuron Output: ", output)
print("----------------------------------")


