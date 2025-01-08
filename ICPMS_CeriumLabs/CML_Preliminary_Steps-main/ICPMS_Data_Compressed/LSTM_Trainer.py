import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

####################################
######      Prepare Data      ######
####################################

data_files = ["ICPMS_Data_0ppt_COLD_processed.csv","ICPMS_Data_0ppt_HOT_processed.csv","ICPMS_Data_200ppt_COLD_processed.csv","ICPMS_Data_200ppt_HOT_processed.csv",
              "ICPMS_Data_050ppt_COLD_processed.csv","ICPMS_Data_500ppt_HOT_processed.csv","ICPMS_Data_1000ppt_COLD_processed.csv","ICPMS_Data_1000ppt_HOT_processed.csv"]

# Load the CSV file of your choosing
data = pd.read_csv(data_files[1])

# Ensure the 'Datetime' column is in datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Sort the dataframe by the 'Datetime' column
data = data.sort_values(by='Datetime')

# Select only Rh103(LR)_AVG values and their associated Datetime
data = data[["Datetime","Rh103(LR)_AVG"]]

# Get unique years in data
years = data['Datetime'].dt.year.unique()

# Finds when the data switches from year to the next. This is used to switch between training & testing.
condition = (data['Datetime'].dt.year == 2023) & (data['Datetime'].dt.year.shift(1) == 2022)
index_switch = data[condition].index[0]
print(f"Index switch is: {index_switch}")

# If you have a GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

####################################
######      Inspect Data      ######
####################################

for year in years:
    df_year = data[data['Datetime'].dt.year == year]
    plt.plot(df_year['Datetime'], df_year["Rh103(LR)_AVG"], label=f'{int(year)}')

plt.title('Rh(103) CPS over time')
plt.xlabel('Date')
plt.ylabel('Counts per second [cps]')

plt.show()

####################################
######    Create Dataframe    ######
####################################

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Datetime', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Entry(t-{i})'] = df['Rh103(LR)_AVG'].shift(i)

    df.dropna(inplace=True)

    return df

# How many previous entries to consider
time_segments = 7

# Time shifted dataframe for LSTM. Change "data" to "data_year" to consider a subset of total data.
prepared_df = prepare_dataframe_for_lstm(data, time_segments)

#Convert to numpy array 
prepared_df_as_np = prepared_df.to_numpy()

# Normalize dataframe. You should technically use separate scalers for training data and test data, but oh well.
scaler = MinMaxScaler(feature_range=(-1, 1))
prepared_df_as_np = scaler.fit_transform(prepared_df_as_np)

# Input is the 1:7 entry
X = prepared_df_as_np[:, 1:]

# Need to flip this so the time increments are not backwards
X = dc(np.flip(X, axis=1))

# Target is the 0 entry
Y = prepared_df_as_np[:, 0]

print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Train on 90% of data, test on remaining 10%
# split_index = int(len(X) * 0.90)
split_index = index_switch

# Split data
X_train = X[:split_index]
X_test = X[split_index:]

Y_train = Y[:split_index]
Y_test = Y[split_index:]

# Reshape to match required input shape
X_train = X_train.reshape((-1, time_segments, 1))
X_test = X_test.reshape((-1, time_segments, 1))

Y_train = Y_train.reshape((-1, 1))
Y_test = Y_test.reshape((-1, 1))

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()
X_test = torch.tensor(X_test).float()
Y_test = torch.tensor(Y_test).float()

####################################
######       Train Model      ######
####################################

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 10, 1)
model.to(device)

print(f"Model: {model}")

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    batch_loss_vals = []
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
            batch_loss_vals.append(avg_loss_across_batches)
            # batch_loss_indx += 1

    return batch_loss_vals

    print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

plot_batch_loss_vals = []

for epoch in range(num_epochs):
    epoch_loss_vals = train_one_epoch()
    validate_one_epoch()

    start_index = len(plot_batch_loss_vals)
    plot_batch_loss_vals += epoch_loss_vals
    end_index = len(plot_batch_loss_vals)

    plt.plot(range(start_index, end_index), epoch_loss_vals, label=f'Epoch {epoch + 1}')

# plt.plot(plot_batch_loss_vals)
plt.legend()
plt.ylabel('Loss')

plt.show()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

####################################
######    Inspect Training    ######
####################################

train_predictions = predicted.flatten()

X_train_inverse = np.zeros((X_train.shape[0], time_segments+1))
X_train_inverse[:, 0] = train_predictions
X_train_inverse = scaler.inverse_transform(X_train_inverse)

train_predictions = dc(X_train_inverse[:, 0])

Y_train_inverse = np.zeros((X_train.shape[0], time_segments+1))
Y_train_inverse[:, 0] = Y_train.flatten()
Y_train_inverse = scaler.inverse_transform(Y_train_inverse)

train_target = dc(Y_train_inverse[:, 0])

plt.plot(train_target, label='Actual Trend')
plt.plot(train_predictions, label='Predicted Trend')
plt.title('Train Data')
plt.ylim(0,1.5E6)
plt.xlabel('Date')
plt.ylabel('CPS')
plt.legend()
plt.show()

####################################
######    Inspect Testing     ######
####################################

test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

X_test_inverse = np.zeros((X_test.shape[0], time_segments+1))
X_test_inverse[:, 0] = test_predictions
X_test_inverse = scaler.inverse_transform(X_test_inverse)

test_predictions = dc(X_test_inverse[:, 0])

Y_test_inverse = np.zeros((X_test.shape[0], time_segments+1))
Y_test_inverse[:, 0] = Y_test.flatten()
Y_test_inverse = scaler.inverse_transform(Y_test_inverse)

test_target = dc(Y_test_inverse[:, 0])

plt.plot(test_target, label='Actual Trend')
plt.plot(test_predictions, label='Predicted Trend')
plt.title('Test Data')
plt.ylim(0,1.5E6)
plt.xlabel('Date')
plt.ylabel('CPS')
plt.legend()
plt.show()

mse = mean_squared_error(test_target, test_predictions)
mae = mean_absolute_error(test_target, test_predictions)
rmse = mean_squared_error(test_target, test_predictions, squared=False)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

residuals = test_predictions - test_target

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()
