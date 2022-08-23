import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sklearn
from sklearn.impute import SimpleImputer


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


pd.options.display.max_columns = 99
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.info())


def cleantens(train_data):
    X = train_data
    X.pop('Ticket')
    X.pop('Cabin')
    X.pop('PassengerId')
    X.pop('Name')
    X = X.dropna()
    y = X.pop('Survived')
    non_num_cat = ['Sex', 'Embarked']

    for col in non_num_cat:
        X[col] = X[col].astype('category').cat.codes
    return X, y


X, y = cleantens(train_data)
numpy_input = X.to_numpy()
targets = y.to_numpy()

inputs = torch.from_numpy(numpy_input).type(torch.float32)
targets = torch.from_numpy(targets).type(torch.float32)

dataset = TensorDataset(inputs, targets)

batch_size = 51

train_load = DataLoader(dataset, batch_size, shuffle=True)
train_load = DeviceDataLoader(train_load, device)


class EasyTitanic(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(EasyTitanic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)

    def forward(self, input_set):
        out = torch.tanh((self.linear1(input_set)))
        out = self.batch_norm1(out)
        out = F.relu(self.linear2(out))
        out = self.batch_norm2(out)
        out = torch.sigmoid(self.linear3(out))
        return out


in_size = 7
hid1 = 243
hid2 = 49
out_size = 1

model = EasyTitanic(in_size, hid1, hid2, out_size)
model = model.to(device)


def binacc(outputs, targets):
    preds = torch.round(outputs)
    correct_result_sum = (preds == targets).sum().float()
    acc = correct_result_sum / targets.shape[0]
    return acc


criterion = nn.BCEWithLogitsLoss()


def fit(epochs, lr, train_load, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for batch in train_load:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            acc = binacc(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_acc += acc.item()
            epoch_loss += loss.item()

        print(
            f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_load):.5f} | Acc: {epoch_acc / len(train_load):.3f}')


fit(1000, 0.01, train_load)

def cleantens1(train_data):
    X = train_data
    X.pop('Ticket')
    X.pop('Cabin')
    passengerID = X.pop('PassengerId')
    X.pop('Name')
    non_num_cat = ['Sex', 'Embarked']

    for col in non_num_cat:
        X[col] = X[col].astype('category').cat.codes
    return X, passengerID


sample_sub = pd.read_csv('gender_submission.csv')
# print(sample_sub.info())
# print(test_data.info())
test_data, passengerID = cleantens1(test_data)


na_cols = ['Age','Fare']
imputer = SimpleImputer( strategy='mean').fit(test_data[na_cols])

test_data[na_cols] = imputer.transform(test_data[na_cols])


'''
test_in = torch.from_numpy(cleantens2(test_data).to_numpy()).type(torch.float32)

newout = torch.round(model(test_in))

print(newout)
'''
