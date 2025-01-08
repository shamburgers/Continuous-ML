#Load models
import pandas as pd
import numpy as np
import sys
import torch
from pathlib import Path
import os
main_path = "C:/Users/sohal/ICPMS_CeriumLabs/CML_Preliminary_Steps-main"
sys.path.append(main_path+"/src2") #Code path location

data_path = main_path+"/ICPMS_Data_Compressed"
sys.path.append(data_path) #Data path lcoation

results_path = data_path+"/results"
os.makedirs(results_path, exist_ok=True)
sys.path.append(results_path) #Results path location
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import random
import matplotlib.pyplot as plt
from models import LSTMVAE, SimpleLSTM, Transformer, Ensemble
import utils


import torch
from pathlib import Path



def load_model(model, directory: str, model_name: str):
    # Ensure the directory path is valid
    directory = Path(directory)
    model_type = model_name.split('_')[0]
    model_load_path = directory / model_name
    
    # Check if the specified model file exists
    assert model_load_path.is_file(), f"[Error]: Model file {model_load_path} not found."

    # Load the model
    print(f"[Loading model from]: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path)) 
    
    return model, model_type


min_value, max_value, train_size, batch_size, sequence_length, num_features, all_elements, elements_list, category, quantity, element_name = utils.load_metadata(dataset_path, year)
