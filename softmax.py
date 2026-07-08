import numpy as np 

input=[ [1,2,3,2.5],
        [2,5,-1,2],
        [-1.5,2.7,3.3,-0.8]]


#Un-normalised probabilities
exp_value=np.exp(input - np.max(input,axis=1,keepdims=True))
#Normalising them for each sample
probabilities= exp_value/ np.sum(exp_value,axis=1,keepdims=True)

print(probabilities)
print(np.sum(probabilities,axis=1))
