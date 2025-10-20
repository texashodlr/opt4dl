"""
Find the image of a matrix using row echelon form
Task: Compute the col space of a matrix --> return the basis vectors that span the col space of A

Input:
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_image(matrix))

Output:
# [[1, 2],
#  [4, 5],
#  [7, 8]]

"""

import numpy as np

def matrix_image(A):
	B = A.copy()
	unique_cols = 0
	ref = row_echelon(A)
	r, c = ref.shape
	C = [[None for _ in range(r)] for _ in range(c)]
	# print(f"R: {r}, C: {c}")
	for col in range(c):
		#print(ref[:, col])
		for row in range(r):
			if ref[row,col] == 1:
				unique_cols += 1
				# print(B[:,col])
				for i in range(r):
					# print(f"i: {i}, r: {r}, B: {B[i,col]}")
					val = B[i,col]
					C[i].append(val)
				# print(f"C: {C}")
				# C = np.hstack([C,B[:,col]])
	# image = np.array(C)
	for i in range(len(C)):
		new_list = [item for item in C[i] if item is not None]
		C[i] = new_list
	image = np.array(C)
	return image


def row_echelon(A):
	""" Returning the REF of the Matrix A"""
	
    # If matrix A has no cols or rows,
	# it is already in REF! So we return itself
	r, c = A.shape
	if r == 0 or c ==0:
		return A

    # We search for non-zero element in the first column
	for i in range(len(A)):
		if A[i,0] != 0:
			break
		else:
			# If all elements in the first column are zero,
			# we preform REF on matrix from the second column
			B = row_echelon(A[:, 1:])
			# and then add the first zero-column back
			return np.hstack([A[:,:1], B])
	# If a non-zero element happens not in the first row,
	# we switch rows!
	if i > 0:
		ith_row = A[i].copy()
		A[i] = A[0]
		A[0] = ith_row
	# We divide the first row by first element in it
	A[0] = A[0] / A[0,0]
	# We then subtract all subsequent rows with the first row (as it now has '1' as it's first element)
	# Multiplied by the corresponding element in the first column
	A[1:] -= A[0] * A[1:,0:1]
	
    # we now perform REF on matrix from second row, from second column
	B = row_echelon(A[1:, 1:])
	
    # We then add first row and first (zero) column, and return!
	return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

"""
A = np.array([[4, 7, 3, 8],
              [8, 3, 8, 7],
              [2, 9, 5, 3]], dtype='float')
"""

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_image(A))