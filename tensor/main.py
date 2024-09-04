# %%

import torch
import numpy as np

# %%
x = torch.empty(3,4)
print(type(x))
print(x)

# %%

zeros = torch.zeros(3,4)
print(zeros)

ones = torch.ones(3, 4)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

# %%
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)

# %%
x1 = torch.empty(2,2,3)
print(x1.shape)
print(x1)

empty_as_x1 = torch.empty_like(x1)
print(empty_as_x1.shape)
print(empty_as_x1)

zeros_as_x1 = torch.zeros_like(x1)
print(zeros_as_x1.shape)
print(zeros_as_x1)

ones_like_x = torch.ones_like(x1)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x1)
print(rand_like_x.shape)
print(rand_like_x)

# %%

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)

# %%

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)

# %%
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")































