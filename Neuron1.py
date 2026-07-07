input = [1, 2, 3, 2.5] # here are the 4 inputs 1st 

weights = [ [0.2, 0.8, -0.5, 1.0], # weights for neuron 1, w11, w12, w13, w14
            [0.5, -0.91, 0.26, -0.5], # weights for neuron 2, w21, w22, w23, w24
            [-0.26, -0.27, 0.17, 0.87] # weights of neuron 3, w31, w32, w33, w34
          ]

bias= [ 2,3,0.5]

output=[
            # Neuron 1: 
            input[0]*weights[0][0]+
            input[1]*weights[0][1]+
            input[2]*weights[0][2]+
            input[3]*weights[0][3]+ bias[0],
            
            
            #Neuron 2:
            input[0]*weights[1][0]+
            input[1]*weights[1][1]+
            input[2]*weights[1][2]+
            input[3]*weights[1][3]+bias[1],
            
            #Neuron 3:
            input[0]*weights[2][0]+
            input[1]*weights[2][1]+
            input[2]*weights[2][2]+
            input[3]*weights[2][3]+bias[2],
       ] 

print("Neuron 1 , Neuron 2 , Neuron 3 ")
print("----------------------------")
print(output)

