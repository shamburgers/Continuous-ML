import torch
from pathlib import Path
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os
# import engine, fetch_data
from torch.utils.data import Dataset, DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import fetch_data
import engine
### Saving models, datasets, and metadata
################################################################################################################################################################################

def save_model(model, directory: str = "models", model_type: str = "model",
                model_name: str = "{}_{}.pt"):
                
    #Create directory
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    assert "{}" in model_name, "Model name must contain a '{}' placeholder for unique tag/identification."
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Model name must end with .pth or .pt"
    random.seed(None)
    saved_model_name = model_name.format(model_type, random.randint(100,999999))
    model_save_path = directory / saved_model_name

    #Save the model
    print(f"[Saving model to]: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    
    return model, saved_model_name

def save_dataset(dataset, dataset_name, save_path, year):
    # Create a subdirectory for the specific year
    base_path = os.path.join(save_path, f"Training_with_{str(year)}" )
    os.makedirs(base_path, exist_ok=True)
    
    # Save the dataset to the year-specific folder
    file_path = os.path.join(base_path, f"{dataset_name}_{year}.pt")
    torch.save(dataset, file_path)
    print(f"{dataset_name} saved to {file_path}")


def save_metadata(min_value, max_value, num_features, batch_size, sequence_length, year: int, save_path, all_elements: bool, elements_list, category, train_size, quantity, element_name):
    # Create a subdirectory for the specific year
    base_path = os.path.join(save_path, f"Training_with_{str(year)}")
    os.makedirs(base_path, exist_ok=True)
    
    # Ensure that the tensors are detached, converted to numpy arrays, and cast to float
    min_value_list = [float(val) for val in min_value.detach().cpu().numpy().flatten()]
    max_value_list = [float(val) for val in max_value.detach().cpu().numpy().flatten()]
    
    # Prepare the metadata dictionary with min-max values and other details
    metadata_dict = {
        "year": int(year),
        "category": category,          
        "quantity": quantity,
        "train_size": train_size,
        "element_name": element_name,
        "all_elements": bool(all_elements),
        "elements_list": elements_list,
        "num_features": num_features,
        "min_value": min_value_list,
        "max_value": max_value_list,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }
    
    print(f"Saving metadata for year {year}...")

    # Save the combined metadata to a JSON file
    with open(f"{base_path}/metadata_{year}.json", "w") as json_file:
        json.dump(metadata_dict, json_file)
    
    print(f"Metadata for year {year} saved successfully!")
################################################################################################################################################################################

#Loading models, datasets, and metadata
def load_dataset(dataset_path, quantity, category, year, sequence_length, batch_size, all_elements, element_name):
    # Create a subdirectory for the specific year
    base_path = os.path.join(dataset_path, f"Training_with_{str(year)}")
    os.makedirs(base_path, exist_ok=True)

    # Initialize loaders for train, valid, and test datasets
    train_loader = valid_loader = test_loader = None

    # Load datasets for train, valid, and test
    for dataset_type in ['train', 'valid', 'test']:
        # Construct file path for each dataset type
        file_path = os.path.join(base_path, f"{dataset_type}_dataset_{quantity}_ppt_{category}_{sequence_length}_{batch_size}_all_elements_{all_elements}_{element_name}_{year}.pt")
        
        # Load the DataLoader if the file exists
        if os.path.exists(file_path):
            # Directly load the DataLoader
            if dataset_type == 'train':
                train_loader = torch.load(file_path)
            elif dataset_type == 'valid':
                valid_loader = torch.load(file_path)
            elif dataset_type == 'test':
                test_loader = torch.load(file_path)
        else:
            print(f"Warning: {file_path} does not exist. Please specify the correct path.")

    return train_loader, valid_loader, test_loader

def load_metadata(saved_path, year: int):
    base_path = os.path.join(saved_path, f"Training_with_{str(year)}")
    file_path = f"{base_path}/metadata_{year}.json"
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    with open(file_path, "r") as json_file:
        metadata = json.load(json_file)
    
    # Extract the min and max values from the metadata
    min_value = torch.tensor(metadata["min_value"])
    max_value = torch.tensor(metadata["max_value"])
    batch_size = metadata["batch_size"]
    sequence_length = metadata["sequence_length"]
    num_features = metadata["num_features"]
    elements_list = metadata["elements_list"]
    category = metadata["category"]
    quantity = metadata["quantity"]
    element_name = metadata["element_name"]
    train_size = metadata["train_size"]
    
    # Cast all_elements back to the boolean type
    all_elements = bool(metadata["all_elements"])

    print(f"Metadata for year {year} loaded successfully!")
    return min_value, max_value, train_size, batch_size, sequence_length, num_features, all_elements, elements_list, category, quantity, element_name
################################################################################################################################################################################

# Other utility functions
def calculate_min_max(dataset):

    min_value = torch.min(dataset, dim=0).values
    max_value = torch.max(dataset, dim=0).values

    return min_value, max_value

# def make_predictions(model, data_loader, num_features, element_list, 
#                     predict_element, model_type:str, year:int, 
#                     min_value, max_value):

#     test_loss, preds, true_values = test_step(model, data_loader, device, model_type)
#     plot(preds, true_values, num_features, element_list, predict_element, model_type, year, min_value, max_value) 
#     print(f"Test Loss for year {year}: {test_loss: .6f}")

#     return preds, true_values

import matplotlib.pyplot as plt

def plot(predictions, true_values, num_features, 
        element_list, predict_element:str, model_type:str, year:int, 
        min_value, max_value):

    # Transform back to original cps values (unscaled ones)
    if num_features == 1:
        # Single feature case
        y_pred = predictions[:, 0]
        y_actual = true_values[:, 0]
        print("Y_pred Shape (1 feature):", y_pred.shape)
        min_val = min_value.item()
        max_val = max_value.item()
        
    else:
        try:
            feature_index = element_list.index(predict_element)
        except ValueError:
            raise ValueError(f"{predict_element} not found in element_list")
        print("Predictions Shape:", predictions.shape)  

        if model_type == "LSTMVAE":
            y_pred = predictions[:, 0, feature_index]
        elif model_type in ["SimpleLSTM", "Transformer", "Ensemble"]:
            y_pred = predictions[:, 0, feature_index]
        
        y_actual = true_values[:, 0, feature_index]

        min_val = min_value[feature_index].item()
        max_val = max_value[feature_index].item()
    
    # Scale back to original values
    y_pred = y_pred * (max_val - min_val) + min_val
    y_actual = y_actual * (max_val - min_val) + min_val

    # Plot
    x_indices = range(len(y_pred))
    plt.plot(x_indices, y_pred, label="Predicted")
    plt.plot(x_indices, y_actual, label="Actual")
    plt.title(f"Predicted vs Actual for {predict_element} in {year}")
    plt.xlabel("Time")
    plt.ylabel("CPS")
    plt.legend()
    plt.show()

    return y_pred, y_actual

################################################################################################################################################################################

def yearly_data(data_per_year, batch_size, sequence_length, min_value, max_value):
    yearly_data_loaders = []  # Initialize a list to store the data for each year
    for i, data in enumerate(data_per_year):
        scaled_data = (data - min_value) / (max_value - min_value)
        X, y = fetch_data.data_to_X_y(scaled_data, sequence_length)
        data_loader = DataLoader(fetch_data.TimeSeriesDataset(X, y), batch_size = batch_size, shuffle = False)
        yearly_data_loaders.append(data_loader)

    return yearly_data_loaders  # Return the list containing data for each year

def plot_yearly(model, data_per_year, sequence_length, batch_size, years, model_type:str,
                element_list, predict_element, num_features, min_value, max_value):
    
    yearly_data_loaders = yearly_data(data_per_year, batch_size, sequence_length, min_value, max_value)
    for dataloader, year in zip(yearly_data_loaders, years):
        test_loss, preds, true_values = engine.test_step(model, dataloader, device, model_type)
        print(f"Test Loss for year {year}: {test_loss: .6f}")
        plot(preds, true_values, num_features, element_list, predict_element, model_type, year, min_value, max_value)

