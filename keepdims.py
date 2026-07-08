import numpy as np

A= [  [1,2,3],
      [4,5,6],
      [7,8,9] 
    ]

print(np.sum(A))
print("-----------------------")

print(np.sum(A,axis=0))
print(np.sum(A,axis=0).shape)
print("-----------------------")


print(np.sum(A,axis=1))
print(np.sum(A,axis=1).shape)
print("-----------------------")

print(np.sum(A,axis=0,keepdims=True))
print(np.sum(A,axis=0,keepdims=True).shape)
print("-----------------------")

print(np.sum(A,axis=1,keepdims=True))
print(np.sum(A,axis=1,keepdims=True).shape)
print("-----------------------")
