# %%
import torch
from torch import nn

# %%
input = torch.randn(20, 16, 50, 100) # batch_size, in_channels, h, w

# %%
# pool of square window of kernel_size=3, stride=2

m = nn.MaxPool2d(3, stride=2)
output = m(input)
output.shape

# %%
# pool of non-square window
m = nn.MaxPool2d((3, 2), stride=(2, 1))
output = m(input)
output.shape

