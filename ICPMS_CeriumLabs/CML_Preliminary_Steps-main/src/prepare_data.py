
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

main_path = "c:/Users/sohal/Continuous-Machine-Learning/ICPMS_CeriumLabs/CML_Preliminary_Steps-main"
sys.path.append(main_path+"/src") #Code path location

data_path = main_path+"/ICPMS_Data_Compressed/"
sys.path.append(data_path) #Data path lcoation

#Split the dataset into training and testing and apply scaling (min-max in this case)

def split_data(dataset, train_size):
    if train_size == 0:
        train_data = dataset
        test_data = dataset
    else:
        split_index = int(len(dataset)*train_size)
        train_data = dataset[:split_index]
        test_data = dataset[split_index:]

    #Scale the data so that it gets between 0 and 1
    #Apply the train scaling to test dataset
    min_value = torch.min(train_data, dim=0).values
    max_value = torch.max(train_data, dim=0).values

    scaled_train = (train_data - min_value) / (max_value - min_value)
    scaled_test = (test_data - min_value) / (max_value - min_value)
    

    return scaled_train, scaled_test, min_value, max_value

def data_to_X_y(data, sequence_length):
    ndim = data.ndim

    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    if data.ndim == 1:
        X = np.expand_dims(X, axis = -1)
        y = np.expand_dims(y, axis = -1)

    # Now get back to converting it to torch tensors
    X = torch.tensor(X)
    y = torch.tensor(y)

    return X, y

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def data_setup(dataset, sequence_length, train_size, batch_size):
    print(len(dataset), dataset.shape)
    scaled_train, scaled_test, min_value, max_value = split_data(dataset, train_size)
    scaled_train, scaled_valid, _, _ = split_data(scaled_train, train_size)
    

    X_train, y_train = data_to_X_y(scaled_train, sequence_length, )
    X_valid, y_valid = data_to_X_y(scaled_valid, sequence_length)
    X_test, y_test = data_to_X_y(scaled_test, sequence_length)
    # print("Original Train Shape:", y_train.shape)

    num_features = X_train.shape[-1]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    valid_dataset = TimeSeriesDataset(X_valid, y_valid)
    test_dataset = TimeSeriesDataset(X_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader, num_features, min_value, max_value
