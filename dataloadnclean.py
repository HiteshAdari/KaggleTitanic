import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

pd.options.display.max_columns = 99
original_train_data = pd.read_csv('train.csv')
original_test_data = pd.read_csv('test.csv')
X = original_train_data.drop('Name', axis=1)
X = X.dropna()
y = X.pop('Survived')


X.pop('Ticket')
X.pop('Cabin')
X.pop('PassengerId')
non_num_cat = ['Sex', 'Embarked']

for col in non_num_cat:
    X[col] = X[col].astype('category').cat.codes
'''
for i in X.index:
    if X.loc[i, 'Sex'] == 'male':
        X.loc[i, 'Sex'] = 1
    elif X.loc[i, 'Sex'] == 'female':
        X.loc[i, 'Sex'] = 2

    if X.loc[i, 'Embarked'] == 'S':
        X.loc[i, 'Embarked'] = 3
    elif X.loc[i, 'Embarked'] == 'C':
        X.loc[i, 'Embarked'] = 1
    elif X.loc[i, 'Embarked'] == 'Q':
        X.loc[i, 'Embarked'] = 2
'''



numpy_input = X.to_numpy()
targets = y.to_numpy()

inputs = torch.from_numpy(numpy_input).type(torch.float32)
targets = torch.from_numpy(targets).type(torch.float32)

dataset = TensorDataset(inputs, targets)

val_percent = 0.15

val_size = int(len(inputs) * val_percent)
train_size = len(inputs) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 1
train_load = DataLoader(train_ds, batch_size, shuffle=True)
val_load = DataLoader(val_ds, batch_size, shuffle=True)
print(original_train_data.describe())
print(original_train_data.corr())

