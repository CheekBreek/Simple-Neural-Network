#Neural network with no bias. Goal is to predict the outcome of a vector with three values. 
#Correct output is the first of the 3 values.

import numpy as np

inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])

outputs = np.array([[0], [0], [0], [1], [1], [1]])

#Starting weights
weights = np.array([[.50], [.50], [.50]])

for count in range(100000):
    prediction = np.dot(inputs, weights)                  #Dot product each feature by corresponding weight
    sigmoid = 1 / (1 + np.exp(-prediction))               #Activation function
    error = outputs - sigmoid                             #Calculating error
    delta = error * sigmoid * (1 - sigmoid)               #Cost function
    weights += np.dot(inputs.T, delta)                    #Updating weights
    
# create two new examples to predict                                   
example1 = np.array([[1, 1, 0]])
example2 = np.array([[0, 1, 1]])

print('Should return 1 - ', 1 / (1 + np.exp(-(np.dot(example1, weights)))))
print('Should return 0 - ', 1 / (1 + np.exp(-(np.dot(example2, weights)))))