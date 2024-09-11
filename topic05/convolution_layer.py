# %%
import torch
from torch import nn

# %%
input = torch.randn(20, 16, 50, 100) # batch_size, in_channels, h, w

# %%
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)

output = m(input)
output.shape

# %%
# non-square kernels and unequal stride and with padding

m= nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))

output = m(input)
output.shape
