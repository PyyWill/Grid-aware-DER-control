import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from predict_model import LSTMNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from tqdm import tqdm

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def mean_absolute_percentage_error(real, predict):
    return torch.mean(torch.abs((real - predict) / real)) * 100

# Load and preprocess dataset
data_path = r'C:\Users\11389\OneDrive\桌面\decision-aware-uq-main\grid_aware_optnn\data\power.csv'
dataset = read_csv(data_path, header=0, index_col=0)
values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, n_in=1, n_out=1)
reframed.drop(reframed.columns[[5,6,7]], axis=1, inplace=True)

values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = torch.tensor(train_X).float().unsqueeze(1)
train_y = torch.tensor(train_y).float().unsqueeze(1)
test_X = torch.tensor(test_X).float().unsqueeze(1)
test_y = torch.tensor(test_y).float().unsqueeze(1)

train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(train_dataset, batch_size=72, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=72, shuffle=False)

model = LSTMNet(input_dim=train_X.shape[2], hidden_dim=64)
model.train()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(
    model,
    loader,
    optimizer,
    show_pbar=False
    ):
    model.train()
    total_loss = 0
    count = 0
    pbar = tqdm(loader, desc='Training Epoch', disable=not show_pbar)
    for batch in pbar:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    average_loss = total_loss / count if count > 0 else 0
    return average_loss

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    avg_loss = train_epoch(model, train_loader, optimizer, show_pbar=True)
    print(f'Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_loss:.4f}')

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs)
        actuals.append(targets)
    for i in range(10):
        print(inputs[i], outputs[i], targets[i])
    print(inputs.shape, outputs.shape, targets.shape)
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()

inv_yhat = np.concatenate((predictions, test_X[:, 0, 1:4]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
inv_y = np.concatenate((actuals, test_X[:, 0, 1:4]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mape = mean_absolute_percentage_error(torch.tensor(inv_y), torch.tensor(inv_yhat))

print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape.item())

plt.plot(inv_y, label='Actual', linewidth=1.0, alpha=1.0)
plt.plot(inv_yhat, label='Predicted', linewidth=1.0, alpha=0.8)
plt.legend()
plt.show()
