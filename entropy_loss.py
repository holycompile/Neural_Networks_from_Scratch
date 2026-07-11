import numpy as np 

softmax_outputs = np.array([  [0.7, 0.1, 0.2],
                              [0.1, 0.5, 0.4],
                              [0.02, 0.9, 0.08] ]) # Red , Green, Green
class_targets=[0, 1, 1]

print("Selected Elements:", softmax_outputs[ [0,1,2],class_targets ])
#meaning: softmax_output[ [0,1,2], class_target]
#         softmax_output[ [0,1,2], [0,1,1] ]
'''
from row 0 select element in column 0 = 0.7 #red
from row 1 select element in column 1 = 0.5 #green
from row 2 select element in column 2 = 0.9 #green

hence the output comes as [ 0.7, 0.5, 0.9]
'''
#range(len(softmax_outputs))= range(3)
neg_log= -np.log(softmax_outputs[ range(len(softmax_outputs)), class_targets ])
print("Negetive Log:",neg_log)
average_loss=np.mean(neg_log)
print("Average Loss:",average_loss)