# %%

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# %%
# concat

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6]])
# Конкатенація вздовж 0-го виміру
result = torch.cat((x, y), dim=0)
print(result)

# %%

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6]])
# Стек вздовж нового виміру (вимір 0)
result = torch.stack((x, y), dim=0)
print(result)

# %%
# Трансляція тензорів (tensor broadcasting)

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

# %%

a = torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)

# %%

a = torch.ones(4, 3, 2)

b = a * torch.rand(4, 3)    # розміри повинні збігатися від останнього до першого

c = a * torch.rand(   2, 3) # 3-й та 2-й димми різні

d = a * torch.rand((0, ))   # не може транслювати з порожнім тензором
