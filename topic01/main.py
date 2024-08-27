# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import warnings

# %%
# filter warnings
warnings.filterwarnings('ignore')

# %%
data = pd.read_csv('../datasets/Module_1_Lecture_2_Class_Spaceship_Titanic.csv')

data = data.set_index('PassengerId')

# %%
data.head()

data.info()

# %%
 TARGET = 'Transported'
 
 FEATURES = [col for col in data.columns if col != TARGET]
 
 text_features = ["Cabin", "Name"]
 
 cat_features = [col for col in data.columns if data[col].nunique() < 25 and col not in text_features]
 cont_features = [col for col in data.columns if data[col].nunique() >= 25 and col not in text_features]
                  
print(f'Number of categorical features: {len(cat_features)}')
print('Categorical features:', cat_features, '\\n')
print(f'Number of continuos features: {len(cont_features)}')
print('Continuos features:', cont_features, '\\n')
print(f'Number of text features: {len(text_features)}')
print('Text features:', text_features)

# %%
#Target  distribution (розподіл цільової змінної)
ax = data[TARGET].value_counts().plot(kind='bar', figsize=(8,5))

for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel('value')
    ax.set_ylabel('count')
    
plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()

# %%
#Continuos features  distribution (розподіл числових змінних)

ax = data.loc[:, cont_features].hist(
    figsize=(10,12), 
    grid=False,
    edgecolor='teal',
    linewidth=.4)

for row in ax:
    for col in row:
        for i in col.containers:
            col.bar_label(i)
            col.set_xlabel('value')
            col.set_ylabel('count')

plt.suptitle("Continuous features distribution")

plt.tight_layout()
plt.show()

# %%

service_features = cont_features[1:]

for feature in service_features:
    data[f'used_{feature}'] = data.loc[:, feature].apply(lambda x: 1 if x > 0 else 0)
    
# %%

data.loc[:, cont_features + ['CryoSleep', 'VIP', TARGET]].corr().style.background_gradient()

# %%

imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]

imputer = SimpleImputer(strategy='median')

imputer.fit(data[imputer_cols])

data[imputer_cols] = imputer.transform(data[imputer_cols])

# %%

data['HomePlanet'].fillna('Gallifrey', inplace=True)
data['Destination'].fillna('Skaro', inplace=True)

# %%

data['CryoSleep_is_missing'] = data['CryoSleep'].isna().astype(int)
data['VIP_is_missing'] = data['VIP'].isna().astype(int)

# %%
display(data['CryoSleep'].value_counts())
display(data['VIP'].value_counts())

# %%
data['CryoSleep'].fillna(False, inplace=True)
data['VIP'].fillna(False, inplace=True)

data['CryoSleep'] = data["CryoSleep"].astype(int)
data['VIP'] = data["VIP"].astype(int)

# %%
dummies = pd.get_dummies(data.loc[:, ['HomePlanet', 'Destination']], dtype=int)
dummies

# %%
data = pd.concat([data, dummies], axis=1)

data.drop(columns=['HomePlanet', 'Destination'], inplace=True)

# %%
data[TARGET] = data[TARGET].astype(int)

# %%
data.drop(["Cabin", "Name"], axis=1, inplace=True)

# %%
X = data.drop(TARGET, axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    random_state=42)

# %%
x_train = X_train.T
x_test = X_test.T

y_train = np.expand_dims(y_train.T, 0)
y_test = np.expand_dims(y_test.T, 0)

print('X train size', x_train.shape)
print('X test size', x_test.shape)
print('y train size', y_train.shape)
print('y test size', y_test.shape)





























