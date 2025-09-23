"""
Write a function that multiplies a matrix by a scalar
    and returns the result
"""

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	for r in range(0,len(matrix)):
		for c in range(0, len(matrix[r])):
			#print(matrix[r][c])
			matrix[r][c] *= scalar
	return matrix
    #return result

matrix = [[1, 2], [3, 4]]
scalar = 2
print(scalar_multiply(matrix, scalar))

# Input
# matrix = [[1, 2], [3, 4]], scalar = 2
# output = [[2, 4], [6, 8]]