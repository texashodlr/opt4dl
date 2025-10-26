"""
Task: Implement Cosine Similarity
    In this task, you need to implement a function cosine_similarity(v1, v2) 
    that calculates the cosine similarity between two vectors. 
    Cosine similarity measures the cosine of the angle between two vectors, 
    indicating their directional similarity.

Input:
    v1 and v2: Numpy arrays representing the input vectors.
Output:
    A float representing the cosine similarity, rounded to three decimal places.
Constraints:
    Both input vectors must have the same shape.
    Input vectors cannot be empty or have zero magnitude.

"""

import numpy as np

def cosine_similarity(v1, v2):
	# Implement your code here
	# Vector shape check:
	if v1.shape != v2.shape:
		return
	if v1.size == 0 or v2.size == 0:
		return
	v1_mag = np.linalg.norm(v1)
	v2_mag = np.linalg.norm(v2)
	if v1_mag == 0.0 or v2_mag == 0.0:
		return
	dot_prod = np.dot(v1,v2)
	ans = dot_prod / (v1_mag * v2_mag)
    #print(f"Magnitudes: V1: {v1_mag} V2: {v2_mag}")
	return float(f'{ans:.3f}')


v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(cosine_similarity(v1, v2))