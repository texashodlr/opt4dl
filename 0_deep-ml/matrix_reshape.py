import numpy as np

# a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
# [[1, 2], [3, 4], [5, 6], [7, 8]]

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	b = np.array(a)
	old_shape = b.shape
	if (old_shape == new_shape):
		pass
	elif (old_shape[1] != new_shape[0] or old_shape[0] != new_shape[1]):
		return []
	reshaped_matrix = b.reshape(new_shape)
	return reshaped_matrix.tolist()

print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (2, 4)))