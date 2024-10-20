# import torch
# print(torch.empty(5))
# print(torch.zeros(5))
# a = torch.zeros(5)
# a[0]=1
# print(a)
import torch

# Create a sample matrix (2D tensor)
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Index a specific column, e.g., the second column (index 1)
column_index = 1
column = matrix[:, column_index]

print(column)


import torch

# Create a sample 2D tensor (matrix)
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Perform the max operation along dimension 0 (columns)
max_values, indices = torch.max(matrix, dim=0)

print("Max values:", max_values)
print("Indices of max values:", indices)


import torch

# Create a sample vector
vector = torch.tensor([1, 2, 3])

# Number of times to stack the vector
num_stacks = 4

# Using torch.stack
stacked_matrix_stack = torch.stack([vector] * num_stacks, dim=0)

# Alternatively, using torch.vstack
stacked_matrix_vstack = torch.vstack([vector] * num_stacks)

print("Stacked Matrix using torch.stack:")
print(stacked_matrix_stack)

print("\nStacked Matrix using torch.vstack:")
print(stacked_matrix_vstack)

# Alternatively, using torch.vstack
stacked_matrix_vstack = torch.hstack([vector.unsqueeze(1)] * num_stacks)

print("Stacked Matrix using torch.stack:")
print(stacked_matrix_stack)

print("\nStacked Matrix using torch.hstack:")
print(stacked_matrix_vstack)

def func(v:torch.Tensor):
    return v.unsqueeze(1)

print(func(torch.tensor([1,1,1])))
a = torch.tensor([1,1,1])
print(a.shape)
print(list(range(5,0,-1)))