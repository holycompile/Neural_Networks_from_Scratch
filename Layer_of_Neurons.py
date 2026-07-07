#layer of Neurons
import numpy as np 

input = [1.0, 2.0, 3.0, 2.5]

weight =[      [0.2, 0.8, -0.5, 1],
                [0.5, -0.91, 0.26, -0.5],
                [-0.26, -0.27, 0.17, 0.87]
         ]

bias = [2.0, 3.0, 0.5]


# converting into numpy arrays

input_array=np.array(input)
weight_array=np.array(weight)
bias_array=np.array(bias)

# Calculating the stuff 

output=np.dot(weight_array, input_array) + bias_array
print(output)