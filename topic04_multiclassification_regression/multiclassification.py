# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns

# %%

df = pd.read_csv('./topic04_multiclassification_regression/../datasets/Module_2_Lecture_2_Class_penguins.csv')

df.sample(5, random_state=42)

# %%
df.info()

# %%
df = df.dropna().reset_index(drop=True)

# %%

plt.figure(figsize=(4,3))
ax = sns.countplot(data=df, x='species')

for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
            
plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(4,3))
ax = sns.countplot(data=df, x='island')

for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
            
plt.suptitle("Island feature distribution")

plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(6,6))
sns.pairplot(data=df, hue='species').fig.suptitle('Numeric features distribution', y=1)
plt.show()

# %%
features = ['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

df = df.loc[:, features]

# %%

df.loc[df['species']=='Adelie', 'species']=0
df.loc[df['species']=='Gentoo', 'species']=1
df.loc[df['species']=='Chinstrap', 'species']=2
df = df.apply(pd.to_numeric)

df.head(2)

# %%

X = df.drop('species', axis =1).values
y = df['species'].values

# %%
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%

X_train , X_test , y_train , y_test = train_test_split(X,
                                                       y,
                                                       random_state = 42, 
                                                       test_size =0.33, 
                                                       stratify=y)

# %%
X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train).long()

X_test = torch.Tensor(X_test).float()
y_test = torch.Tensor(y_test).long()

print(X_train[:1])
print(y_train[:10])

# %%

class LinearModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=20, out_dim=3):
        super().__init__()
        
        self.features = nn.Sequential(
            
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )    
        
    def forward(self, x):
        output = self.features(x)
        return output

# %%
model = LinearModel(X_train.shape[1], 20, 3)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epoch = 400 

# %%
train_loss = []
test_loss = []

train_accs = []
test_accs = []

# %%

for epoch in range(num_epoch):
    
    # train the model
    model.train()
    
    outputs = model(X_train)
    
    loss = criterion(outputs, y_train)    
    train_loss.append(loss.cpu().detach().numpy())
    
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()
    
    acc = 100 * torch.sum(y_train==torch.max(outputs.data, 1)[1]).double() / len(y_train)
    train_accs.append(acc)
    
    if (epoch+1) % 10 == 0:
        print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                       %(epoch+1, num_epoch, loss.item(), acc.item()))
        
    # test the model
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        
        loss = criterion(outputs, y_test)
        test_loss.append(loss.cpu().detach().numpy())
        
        acc = 100 * torch.sum(y_test==torch.max(outputs.data, 1)[1]).double() / len(y_test)
        test_accs.append(acc)

# %%
plt.figure(figsize=(4, 3))
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training vs Validation Loss')
plt.show()

# %%
plt.figure(figsize=(4, 3))
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Metric')
plt.show()
































