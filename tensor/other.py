# %%
import torch

tensor = torch.ones(4, 4)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# %%
print(f"{tensor} \\n")
tensor.add_(5)
print(tensor)

# %%

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# %%
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# %%
import numpy as np

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# %%
# Переносимо наш тензор на графічний процесор (GPU), якщо він доступний
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print('here')


