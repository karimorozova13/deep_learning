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
data = pd.read_csv('./datasets/Module_1_Lecture_2_Class_Spaceship_Titanic.csv')

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

# %%

def initialize_weights_bias(dimension):
    # dimension - number of input features
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w,b

# %%

# return y_pred
def sigmoid(z):
    y_pred = 1 / (1 + np.exp(-z))
    return y_pred

# %%

def forward_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    
    # probabilistic 0-1
    y_pred = sigmoid(z)
    
    loss = -1 * y_train * np.log(y_pred) - (1 - y_train) * np.log(1 - y_pred)
    
    # x_train.shape[1]  is for scaling
    cost = np.sum(loss) / x_train.shape[1]
    
    return cost

# %%
# похідні 
derivative_weight = np.dot(x_train, (y_pred - y_train).T) / x_train.shape[1]

derivative_bias = np.sum(y_pred - y_train) / x_train.shape[1]

# %%

def forward_backward_propagation(w, b, x_train, y_train, eps=1e-5):
    
    #forward propagation
    z = np.dot(w.T, x_train) + b
    y_pred = sigmoid(z)
    loss = -1 * y_train * np.log(y_pred + eps) - (1 - y_train) * np.log(1 - y_pred + eps)
    cost = np.sum(loss) / x_train.shape[1]
    
    # backward propagation
    derivative_weight = np.dot(x_train, (y_pred - y_train).T) / x_train.shape[1]
    derivative_bias = np.sum(y_pred - y_train) / x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost, gradients

# %%

#updating (learning) parameters

def update(w, b, x_train, y_train, learning_rate, num_of_iter):
    cost_list = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion 
    for i in range(num_of_iter):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        index.append(i)
        
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        
    # we update(learn) parameters weights and bias
    parameters = {'weight': w, 'bias': b}
    plt.plot(index, cost_list)
    plt.xticks(index, rotation='vertical')
    plt.xlabel('Number of iteration')
    plt.ylabel('Cost')
    plt.show()
    
    return parameters, gradients, cost_list

# %%

def predict(w, b, x_test):

    z = sigmoid(np.dot(w.T, x_test) + b)

    Y_pred  = np.zeros((1,x_test.shape[1]))
    
    # if z > 0.5, our prediction is 1 (y_pred=1)
    # if z <= 0.5, our prediction is 0 (y_pred=0)
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_pred[0,i] = 0
        else:
            Y_pred[0,i] = 1

    return Y_pred        

# %%

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_of_iter):
    dimension = x_train.shape[0]
    w, b = initialize_weights_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_of_iter)
    
    y_pred_test = predict(parameters['weight'], parameters['bias'], x_test)
    y_pred_train = predict(parameters['weight'], parameters['bias'], x_train)
    
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))
    
# %%

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.00001, num_of_iter=50)

























