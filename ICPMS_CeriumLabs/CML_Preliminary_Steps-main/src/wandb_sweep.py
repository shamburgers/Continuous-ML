
import torch
import sys
import numpy as np
import random
import wandb 
import random
import copy


main_path = "c:/Users/sohal/Continuous-Machine-Learning/ICPMS_CeriumLabs/CML_Preliminary_Steps-main"
sys.path.append(main_path+"/src") #Code path location

data_path = main_path+"/ICPMS_Data_Compressed"
sys.path.append(data_path) #Data path lcoation

from models import LSTMVAE, SimpleLSTM, Transformer, Ensemble
import fetch_data, models, prepare_data, engine, utils

device = "cuda" if torch.cuda.is_available() else "cpu"


wandb.login()
# number = random.randint(0, 9999)
# wandb.init(project="project", name=f"test_{number}")

#Uncomment the following to run a full range of hyperparameters of your preference
#This sweep_configuration should be changed everytime you go from one task to another.
#Documentation @ https://docs.wandb.ai/guides/sweeps
# sweep_configuration = {
#     "method": "grid",   #Other options are ["grid", "Bayes"]
#     "name": "sweep",
#     "metric": {"goal": "minimize", "name": "val_loss"},
#     "parameters": {
#         "quantity":{
#             "values": [0]
#         },
#         "category": {
#             "values": ["HOT"]
#         },
#         "year": {
#             "values": [2022]
#         },
#         "predict_other_year" : {
#             "values": [2023]
#         },
#         "all_elements": {
#             "values": [False]
#         },
#         "element_name": {
#             "values": ["Rh103(LR)"]
#         },
#         "model_type": {
#             "values": ["SimpleLSTM"]
#         },
#         "predict_element": {
#             "values": ["Rh(103)"]
#         },
#         "sequence_length": {
#             "values": [2, 3, 4, 5, 6, 7]
#             },
#         "hidden_size": {
#             "values": [32, 64, 128]
#         },
#         "latent_size": {
#             "values": [2, 4, 8, 16, 32, 64]
#         },
#         "num_layers": {
#             "values": [1, 2, 3, 4, 5]
#         },
#         "num_epochs": {
#             "values": [50, 100, 150]
#         },
#         "batch_size": {
#             "values": [2, 3, 4, 5, 6, 7, 8] # Incremental values for batch size
#         },
#         "learning_rate": {
#             "values" : [0.01, 0.001]
#         },
#         "kld_weight": {
#             "values": [0.000025]
#         },
#         "d_model": {
#             "values": [64, 128, 256]
#         },
#         "num_heads": {
#             "values": [2, 4, 6, 8]
#         },
#         "transformer_num_encoder_layers": {
#             "values": [2, 3, 4, 5]
#         },
#         "dim_feedforward": {
#             "values": [64, 128, 256]
#         },
#         "dropout": {
#             "values": [0.1, 0.2, 0.3]
#         },
#         "train_size": {
#             "values": [0.6, 0.7, 0.8]
#         },
#     },
# }

#Comment out the configuration below out after uncommenting the one above
sweep_configuration = {
    "method": "grid",   #Other options are ["grid", "Bayes"]
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "quantity":{
            "values": [0]
        },
        "category": {
            "values": ["HOT"]
        },
        "year": {
            "values": [2022]
        },
        "all_elements": {
            "values": [False]
        },
        "element_name": {
            "values": ["Rh103(LR)"]
        },
        "model_type": {
            "values": ["SimpleLSTM"]
        },
        "predict_element": {
            "values": ["Rh(103)"]
        },
        "sequence_length": {
            "values": [2, 3]
            },
        "hidden_size": {
            "values": [64]
        },
        "latent_size": {
            "values": [8]
        },
        "num_layers": {
            "values": [2]
        },
        "num_epochs": {
            "values": [120]
        },
        "batch_size": {
            "values": [3] # Incremental values for batch size
        },
        "initial_learning_rate": {
            "values" : [0.001]
        },
        "kld_weight": {
            "values": [0.000025]
        },
        "d_model": {
            "values": [64]
        },
        "num_heads": {
            "values": [8]
        },
        "transformer_num_encoder_layers": {
            "values": [3]
        },
        "dim_feedforward": {
            "values": [256]
        },
        "dropout": {
            "values": [0.1]
        },
        "train_size": {
            "values": [0.8]
        },
    },
}

def set_seed(seed=42):
    print('Setting seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(1000000)

def model_training():
    run = wandb.init(project="project", config=sweep_configuration)
    
    # Set a unique run name based on the hyperparameters
    run.name = f"{run.config.model_type}_trainsize_{run.config.train_size}_timewindow_{run.config.sequence_length}_init.lr_{run.config.initial_learning_rate}_bs_{run.config.batch_size}_hs_{run.config.hidden_size}_num.layers_{run.config.num_layers}_kldweight_{run.config.kld_weight}_latentsize_{run.config.latent_size}"
    
    config = wandb.config

    # Extract parameters from the config dictionary
    quantity = config["quantity"]
    category = config["category"]
    year = config["year"]
    all_elements = config["all_elements"]
    element_name = config["element_name"]
    sequence_length = config["sequence_length"]
    # input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    latent_size = config["latent_size"]
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["initial_learning_rate"]
    kld_weight = config["kld_weight"]
    train_size = config["train_size"]
    model_type = config["model_type"]
    predict_element = config["predict_element"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    transformer_num_encoder_layers = config["transformer_num_encoder_layers"]
    dim_feedforward = config["dim_feedforward"]
    dropout = config["dropout"]

    dataset, elements_list, data_per_year, indices, years = fetch_data.select_data(quantity = quantity,
                                    category = category,
                                    year = year,
                                    all_elements = all_elements,
                                    element_name = element_name)

    train_loader, valid_loader, test_loader, num_features, min_value, max_value = prepare_data.data_setup(
        dataset = dataset,
        sequence_length=sequence_length,
        train_size=train_size,
        batch_size=batch_size,
    )    


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
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=num_epochs,
        scheduler=lr_scheduler,
        device=device,
        model_type=model_type,
        # early_stop_patience=20
    )

    # Finish wandb run
    wandb.finish()

    #Save the model
    model, model_name  = utils.save_model(model = model, model_type = model_type)
    print("MODEL BEING USED:", model_name)

    #Make predictions and plot
    print(f"Here are the predictions")
    utils.plot_yearly(model = model, data_per_year = data_per_year, sequence_length = sequence_length, 
                    batch_size = batch_size, years = years, model_type = model_type, 
                    element_list = elements_list, predict_element = predict_element, num_features = num_features)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="ICPMS")
    wandb.agent(sweep_id, function=model_training)
