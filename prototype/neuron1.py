inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2.0
sum=0.0
for i in range(len(inputs)):
    output = inputs[i] * weights[i] 
    sum=sum+output
    print(f"Input: {inputs[i]}, Weight: {weights[i]}, Bias: {bias}, Output: {output}")
print(sum+bias)