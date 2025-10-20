"""
Write a python function to convert a 1D numpy array into a diagonal matrix.
The function should take in a 1D numpy array x and return a 2D numpy array representing the diagonal matrix.
In
x = np.array([1, 2, 3])
    output = make_diagonal(x)
    print(output)
Out:
	[[1. 0. 0.]
    [0. 2. 0.]
    [0. 0. 3.]]
	
	Basically 1xN --> NxN Array where the n'th Item in the NxN 
"""

import numpy as np

def make_diagonal(x):
	dim = len(x)
	zero_matrix = np.zeros((dim, dim))
	for n in range(dim+1):
		zero_matrix[n-1, n-1] = x[n-1]
	print(zero_matrix)
	return zero_matrix

x = np.array([4, 4, 4])
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
output = make_diagonal(x)
print(output)
print(make_diagonal(np.array([4, 4, 4])))