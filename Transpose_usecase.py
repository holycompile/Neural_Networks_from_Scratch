import numpy as np 

inputs = [  [1.0, 2.0, 3.0, 2.5], 
            [2.0, 5.0, -1.0, 2.0], 
            [-1.5, 2.7, 3.3, -0.8]
          ]  # this is a 3 X 4 MATRIX 

weights = [ [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]  ]  # this is a 3 x 4 MATRIX 

biases = [ 2.0, 3.0, 0.5 ]

#converting into numpy arrays 

input_array= np.array(inputs) # 3*4
weight_array=np.array(weights) # 3*4
bias_array=np.array(biases)

#using the dot product 
# note for dot product we need ( 3 * 4 ) and ( 4 * 3 )
output = np.dot(input_array, weight_array.T)+bias_array

print(output)