#Q1:
"""

Write a Python function that computes the dot product of a matrix and a vector.
The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. 
A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector.
For example, an n x m matrix requires a vector of length m.

"""
def matrix_dot_product(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    result_vec = []
    if len(b) != len(a):
        return -1
    else:
        
        for row in a:
            sum = 0
            col_count = 0
            for col in b:
                #print(f"Row: {row} | Col Count: {col_count}")
                #print(f"Col: {col} | Row Element: {row[col_count]}")
                sum += col * row[col_count]
                col_count += 1
            result_vec.append(sum)
        return result_vec


a = [[1, 2], [2, 4]]
b = [1, 2]

vec = matrix_dot_product(a,b)
print(vec)

a = [[1, 2, 3], [2, 4, 5], [6, 8, 9]]
b = [1, 2, 3]

vec = matrix_dot_product(a,b)
print(vec)
        
