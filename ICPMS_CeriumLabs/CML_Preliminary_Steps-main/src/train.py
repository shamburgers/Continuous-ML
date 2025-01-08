
import torch
import sys
import numpy as np
import random
import wandb 
import random
import copy
import os
from torch.utils.data import Dataset
import argparse

main_path = "c:/Users/sohal/Downloads/CML_Preliminary_Steps-main/CML_Preliminary_Steps-main"
sys.path.append(main_path+"/src2") #Code path location

data_path = main_path+"/ICPMS_Data_Compressed"
sys.path.append(data_path) #Data path lcoation

# from src2 import fetch_data, models, engine, utils
import fetch_data
from fetch_data import TimeSeriesDataset
from models import LSTMVAE, SimpleLSTM, Transformer, Ensemble
import utils
from utils import load_metadata, load_dataset, save_model, plot_yearly
import engine

device = "cuda" if torch.cuda.is_available() else "cpu"



number = random.randint(0, 9999)
# wandb.init(project="project", name=f"test_{number}")
def set_seed(seed=42):
    print('Setting seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    


        

def model_training(config):
    # Extract parameters from the config dictionary
    dataset_path = config["dataset_path"]
    year = config["year"]
    predict_element = config["predict_element"]
    hidden_size = config["hidden_size"]
    latent_size = config["latent_size"]
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    kld_weight = config["kld_weight"]
    model_type = config["model_type"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    transformer_num_encoder_layers = config["transformer_num_encoder_layers"]
    dim_feedforward = config["dim_feedforward"]
    dropout = config["dropout"]
    



    #Load the metadata
    min_value, max_value, train_size, batch_size, sequence_length, num_features, all_elements, elements_list, category, quantity, element_name = utils.load_metadata(dataset_path, year)

    #Other important objects for plotting
    _, _, data_per_year, _, years = fetch_data.select_data(quantity = quantity, category=category, year = year, all_elements = all_elements, element_name = element_name)

    #Load the dataset
    train_loader, val_loader, test_loader = utils.load_dataset(dataset_path, 
                                                     quantity, 
                                                     category, 
                                                     year, 
                                                     sequence_length, 
                                                     batch_size, 
                                                     all_elements, 
                                                     element_name)

    # set_seed(999999)
    set_seed(99999)

    # Initialize models
    model_class = {
        "LSTMVAE": LSTMVAE,
        "SimpleLSTM": SimpleLSTM,
        "Transformer": Transformer,
        "Ensemble": Ensemble
    }.get(model_type)
    
    if model_class is None:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
    
    if model_type == "LSTMVAE":
        model = model_class(
            input_size=num_features,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            kld_weight=kld_weight,
            device=device
        ).to(device)
    elif model_type == "SimpleLSTM":
        model = model_class(
            input_size=num_features,
            hidden_size=hidden_size,
            num_stacked_layers=num_layers
        ).to(device)
    elif model_type == "Transformer":
        model = model_class(
            input_size=num_features,
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=transformer_num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)
    elif model_type == "Ensemble":
        model = model_class(
            input_size = num_features, 
            hidden_size = hidden_size, 
            latent_size = latent_size, 
            kld_weight = kld_weight, 
            num_layers = num_layers,
            d_model = d_model, 
            nhead = num_heads, 
            num_encoder_layers = transformer_num_encoder_layers, 
            dim_feedforward = dim_feedforward).to(device)


    #Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Train the model
    model = engine.train(
        model=model, 
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=num_epochs,
        scheduler=lr_scheduler,
        device=device,
        model_type=model_type,
        # early_stop_patience=20
    )

    #Save the model
    model, model_name  = utils.save_model(model = model, model_type = model_type)
    print("MODEL BEING USED:", model_name)

    # #Make predictions and plot
    print(f"Here are the predictions")
    utils.plot_yearly(model = model,  data_per_year= data_per_year, sequence_length = sequence_length, 
                    batch_size = batch_size, years = years, model_type = model_type, 
                    element_list = elements_list, predict_element = predict_element, num_features = num_features,
                    min_value = min_value, max_value = max_value)

    # loader, _, _, num_features, min_value, max_value = prepare_data.data_setup(
    #     dataset = dataset,
    #     sequence_length=sequence_length,
    #     train_size=0,
    #     batch_size=batch_size,
    # )

    # preds_year, actual_year = utils.make_predictions(model = model,
    #                         data_loader = loader,
    #                         num_features = num_features,
    #                         element_list = elements_list,
    #                         predict_element = predict_element,
    #                         model_type = model_type,
    #                         min_value = min_value,
    #                         max_value = max_value,
    #                         year = predict_other_year)
def main():
    parser = argparse.ArgumentParser(description="Train Time Series Model")
    parser.add_argument('--dataset_path', type=str, required = True, help="Path to the dataset")
    parser.add_argument('--year', type=int, required = True, help="Year of the data")
    parser.add_argument('--hidden_size', type=int, help="Hidden size for the model")
    parser.add_argument('--latent_size', type=int, help="Latent size for the model")
    parser.add_argument('--num_layers', type=int, help="Number of layers in the model")
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training")
    parser.add_argument('--predict_element', type=str, help="Element to predict")
    # parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for the optimizer")
    parser.add_argument('--kld_weight', type=float, help="KLD weight for the LSTM model")
    parser.add_argument('--model_type', type=str, choices=["LSTMVAE", "SimpleLSTM", "Transformer", "Ensemble"], help="Type of model to use")
    parser.add_argument('--d_model', type=int, help="d_model for the transformer")
    parser.add_argument('--num_heads', type=int, help="Number of heads for the transformer")
    parser.add_argument('--transformer_num_encoder_layers', type=int, help="Number of encoder layers in transformer")
    parser.add_argument('--dim_feedforward', type=int, help="Feedforward dimension in the simple transformer model")
    parser.add_argument('--dropout', type=float, help="Dropout rate for the model")
    
    args = parser.parse_args()


    # Setting device
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility

    # Convert args to a dictionary for model training
    config = vars(args)

    # Call the model training function
    model_training(config)

if __name__ == "__main__":
    main()
