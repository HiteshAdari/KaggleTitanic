import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloadnclean import train_load, val_load, train_size, val_size

'''
def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))


class TitanicNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, out_size)

    def forward(self, inputs):
        out = torch.sigmoid(self.linear1(inputs))
        return out

    def training_step(self, batch):
        inputs, targets = batch
        output = self(inputs)
        loss = F.cross_entropy(output, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        output = self(inputs)
        loss = nn.BCELoss(output,targets)
        acc = accuracy(output, targets)
        return {'val loss': loss, 'val acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_loss = [x['val loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        return {'val loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
'''


def accuracy(preds, targets):
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))


class TitanicNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TitanicNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out

    def batch_loss(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets.unsqueeze(1))
        return loss

    def validation_test(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        acc_outputs = torch.round(outputs)
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets.unsqueeze(1))
        acc = accuracy(acc_outputs, targets)
        return {'val_loss': loss, 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result1):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result1['val_loss'], result1['val_acc']))


def evaluate(model, val_loader):
    outputs = [model.validation_test(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_load, val_load, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []
    for epoch in range(epochs):
        for batch in train_load:
            optimizer.zero_grad()
            loss = model.batch_loss(batch)
            loss.backward()
            optimizer.step()

        result1 = evaluate(model, val_load)
        model.epoch_end(epoch, result1)
        history.append(result1)

    return history


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = 'cpu'#get_default_device()


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


train_load = DeviceDataLoader(train_load, device)
val_load = DeviceDataLoader(val_load, device)

model = TitanicNN(7, 15, 1).to('cpu')

fit(200, 0.00001, model, train_load, val_load)

