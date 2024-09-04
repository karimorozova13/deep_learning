# %%
import torch 

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

# %%

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)

# %%

a = torch.ones(4, 3, 2)

c = a * torch.rand(3, 1) # 3-й дим = 1, 2-й дим ідентичний до a
print(c)

# %%

a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # спроба помножити a * b призведе до помилки під час виконання
c = b.unsqueeze(1)       # змінити на двовимірний тензор, додавши в кінці новий dim
print(c.shape)
print(a * c)             # трансляція знову працює!

# %%

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# також може викликати його як метод у модулі torch:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
