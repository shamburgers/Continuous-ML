import pandas as pd
import numpy as np
import sys
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
# from utils import save_dataset, save_metadata, load_metadata, calculate_min_max
import utils
main_path = "c:/Users/sohal/Downloads/CML_Preliminary_Steps-main/CML_Preliminary_Steps-main"
sys.path.append(main_path+"/src2") #Code path location

data_path = main_path+"/ICPMS_Data_Compressed"
sys.path.append(data_path) #Data path lcoation

results_path = data_path+"/results"
sys.path.append(results_path) #Results path location



def select_data(quantity: int,
                category: str,  
                year: int, 
                all_elements: bool = False, 
                element_name: str = None):
    
    # Load the CSV file of your choosing
    data = pd.read_csv(f"{data_path}/ICPMS_Data_{quantity}ppt_{category}_processed.csv")

    # Ensure the 'Datetime' column is in datetime format
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # Sort the dataframe by the 'Datetime' column
    data = data.sort_values(by='Datetime')
    years = data['Datetime'].dt.year.unique()

    # Filter the data by the chosen year   
    print("Available years:", years) 
    print("Chosen year:", year)
    data_year = data[data['Datetime'].dt.year == year]

    # Use an OrderedDict to preserve the order of elements
    elements_ordered = OrderedDict()
    keys = data_year.keys()

    for key in keys:
        # Split the key by underscores and get the element part
        parts = key.split('_')
        if len(parts) > 1:
            element = parts[0]
            elements_ordered[element] = None  # Adding element as a key to preserve order

    elements_list = list(elements_ordered.keys())

    indices = []
    data_per_year = []
    
    for i in years:
        condition = data['Datetime'].dt.year == i
        if all_elements:
            column_average = data_year[[f"{element}_AVG" for element in elements_list]]
            data_per_year.append(torch.tensor(data.loc[condition, [f"{element}_AVG" for element in elements_list]].to_numpy()).float())
        else:
            if element_name:
                column_average = data_year[f"{element_name}_AVG"]
                data_per_year.append(torch.tensor(data.loc[condition, f"{element_name}_AVG"].to_numpy()).float())
            else:
                raise ValueError("Element must be provided")

        indices.append(data[condition].index.to_numpy())

    # Only choose average values
    new_data_chosen_year = torch.tensor(column_average.to_numpy()).float()

    print(column_average)
    return new_data_chosen_year, elements_list, data_per_year, indices, years

def split_data(dataset, train_size):
    
    train_data, test_data = train_test_split(dataset, train_size=train_size, random_state=999999, shuffle=False)
    train_data, valid_data = train_test_split(train_data, train_size=train_size, random_state=999999, shuffle=False)

    return train_data, valid_data, test_data


def scaling_data(train_data, valid_data, test_data, min_value, max_value):

    train_data_scaled = (train_data - min_value) / (max_value - min_value)
    valid_data_scaled = (valid_data - min_value) / (max_value - min_value)
    test_data_scaled = (test_data - min_value) / (max_value - min_value)

    return train_data_scaled, valid_data_scaled, test_data_scaled


def create_data_loaders(train_data, valid_data, test_data, sequence_length, batch_size):
    X_train, y_train = data_to_X_y(train_data, sequence_length)
    X_valid, y_valid = data_to_X_y(valid_data, sequence_length)
    X_test, y_test = data_to_X_y(test_data, sequence_length)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(TimeSeriesDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader

def data_to_X_y(data, sequence_length):

    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+1:i+sequence_length+1])
        # max_val = np.max(data[i:i+sequence_length])
        # min_val = np.min(data[i:i+sequence_length])
        # X.append((data[i:i+sequence_length] - min_val) / (max_val - min_val))
        # y.append((data[i+1:i+sequence_length+1] - min_val) / (max_val - min_val))

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


def data_setup(quantity, 
         category, 
         year,
         scaling_year,
         all_elements,
         element_name,
         sequence_length,
         train_size,
         batch_size,
         save_path):

    # category = category
    # quantity = quantity
    # element_name = element_name
    # train_size = train_size


    dataset, elements_list, data_per_year, indices, years = select_data(quantity, category, year, all_elements, element_name)
    print("Dataset:", dataset)
    print("Element_list:", elements_list)
    print("Data per year:", data_per_year)
    print("Indices:", indices)
    print("Years Available:", years)

    min_value, max_value = None, None  # Initialize min and max values
    # save_dataset(data_per_year, "dataset_per_year", save_path= save_path, year = year)

    for y, d in zip(years, data_per_year):
        print(f"Year: {y}, Data Shape: {d.shape}")

        # Split the data first
        train_data, valid_data, test_data = split_data(d, train_size)
        num_features = train_data.shape[1] if train_data.ndim > 1 else 1  # Number of features


        if y == scaling_year:
            # Calculate min-max for the training data of the scaling year
            min_value, max_value = utils.calculate_min_max(train_data)
            
            # Scale the training, validation, and test data
            train_data_scaled, valid_data_scaled, test_data_scaled = scaling_data(train_data, valid_data, test_data, min_value, max_value)

            # Create data loaders
            train_loader, valid_loader, test_loader = create_data_loaders(train_data_scaled, valid_data_scaled, test_data_scaled, sequence_length, batch_size)

        else:
            min_value, max_value, _, _, _, _, _, _, _, _, _= utils.load_metadata(save_path, scaling_year)
            # Use min-max values from the scaling year for other years
            # if min_value is None or max_value is None:
            #     raise ValueError("Min and Max values must be defined from the scaling year before processing other years.")
            
            # Scale the training, validation, and test data using base year's min-max values
            train_data_scaled, valid_data_scaled, test_data_scaled = scaling_data(train_data, valid_data, test_data, min_value, max_value)

            # Create data loaders
            train_loader, valid_loader, test_loader = create_data_loaders(train_data_scaled, valid_data_scaled, test_data_scaled, sequence_length, batch_size)

        # Save datasets for the current year
        utils.save_dataset(train_loader, f"train_dataset_{quantity}_ppt_{category}_{sequence_length}_{batch_size}_all_elements_{all_elements}_{element_name}", save_path, year=y)
        utils.save_dataset(valid_loader, f"valid_dataset_{quantity}_ppt_{category}_{sequence_length}_{batch_size}_all_elements_{all_elements}_{element_name}", save_path, year=y)
        utils.save_dataset(test_loader, f"test_dataset_{quantity}_ppt_{category}_{sequence_length}_{batch_size}_all_elements_{all_elements}_{element_name}", save_path, year=y)

        # Save metadata for the processed year only
        utils.save_metadata(min_value, max_value, num_features, batch_size, sequence_length, y, save_path, all_elements, elements_list, category, train_size, quantity, element_name)
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and save time series datasets for modeling.")

    # Adding arguments for the function parameters
    parser.add_argument('--quantity', type=int, required=True, help='Related to ppt')
    parser.add_argument('--category', type=str, required=True, help='This can either be HOT or COLD')
    parser.add_argument('--year', type=int, required=True, help='Year of data to be considered')
    parser.add_argument('--sequence_length', type=int, default = 2, help='Time window size to consider')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training set size fraction')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for data loaders')
    parser.add_argument('--all_elements', action='store_true', help='Whether to use all elements or not')
    parser.add_argument('--element_name', type=str, default=None, help='Specific element to focus on of all_elements is False')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the datasets')
    parser.add_argument('--scaling_year', type=int, default=None, help='Year to select for min-max scaling')

    args = parser.parse_args()

    # Call main with the parsed arguments
    # Call main with the parsed arguments
    data_setup(
        quantity=args.quantity,
        category=args.category,
        year=args.year,
        sequence_length=args.sequence_length,
        train_size=args.train_size,
        batch_size=args.batch_size,
        all_elements=args.all_elements,
        element_name=args.element_name,
        save_path=args.save_path,
        scaling_year = args.scaling_year
    )   