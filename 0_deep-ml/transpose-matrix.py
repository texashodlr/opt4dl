# Write a Python function that computes the transpose of a given matrix.
"""
Example:
Input:
a = [[1,2,3],[4,5,6]]
Output:
[[1,4],[2,5],[3,6]]

"""

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    #print(f"Initial Rows: {len(a)} | Initial Cols: {len(a[0])}")
    if not a:
        return -1
    else:
        init_rows = len(a)
        init_columns = len(a[0])
        t_pose = []
        count = 0
        for c in range(init_columns):
            column = []
            for r in range(init_rows):
                column.append(a[r][c])
            t_pose.append(column)
        return t_pose


#a = [[1,2,3],[4,5,6]]
a = [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]

answer = transpose_matrix(a)
print(answer)
"""
[0][0]
[1][0]
[0][1]
[1][1]
[0][2]
[1][2]
"""