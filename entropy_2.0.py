#HOT CODED 
import numpy as np 

y_check_true=np.array([  [1,0,0],
                         [0,1,0],
                         [0,1,0]  ]) # Red, Green, Green
y_pred_clipped_check=np.array([ [0.7, 0.2, 0.1],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08] ])
A= y_check_true*y_pred_clipped_check
'''
A= now looks like, 
[   [0.7, 0 , 0 ],
    [ 0, 0.5, 0 ],
    [ 0, 0.9, 0 ]
]
'''
print(A)
print(A.shape)
print("-----------------------")
B= np.sum(A,axis=1)
'''
B now looks like = 
[[0.7]
 [0.5]
 [0.9] 
]
'''
print(B)
print(B.shape)
print("--------------------")

C= -np.log(B) #natural log (ln)
'''
C now looks like = 
[[0.35] # -ln(0.7)
 [0.69] # -ln(0.5)
 [0.10] # -ln(0.9)
]
'''
print(C)
print(C.shape)
print("----------------")
print(np.mean(C))