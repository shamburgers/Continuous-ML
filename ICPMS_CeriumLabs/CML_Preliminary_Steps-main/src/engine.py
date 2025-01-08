
import torch
import sys
import wandb
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np

# wandb.login()


import random
import copy

number = random.randint(0, 9999)
# wandb.init(project="project", name=f"test_{number}")
# print(f"Test Number: test_{number}")


main_path = "c:/Users/sohal/Downloads/CML_Preliminary_Steps-main/CML_Preliminary_Steps-main"
sys.path.append(main_path+"/src2") #Code path location

data_path = main_path+"/ICPMS_Data_Compressed"
sys.path.append(data_path) #Data path lcoation

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model, data_loader, optimizer, device, model_type:str, clip = 0.3):
    
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        

        if model_type == "LSTMVAE":
            loss, y_pred, _ = model(X)
        elif model_type in ["SimpleLSTM", "Transformer", "Ensemble"]:
            y_pred = model(X)
            loss_function = nn.MSELoss() #Change loss function here if desired
            loss = loss_function(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        train_loss += loss.item()
        
        optimizer.step()

        #Optional: Log the loss for each batch
        # wandb.log({"batch_train_loss": loss.item(), "epoch": epoch})

    train_loss = train_loss / len(data_loader)
    return train_loss


def test_step(model, data_loader, device, model_type: str, step_name="test"):
    model.eval()
    test_loss = 0
    preds = []
    true_values = []
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            true_values.append(y.cpu().numpy())
            
            if model_type == "LSTMVAE":
                loss, y_pred, _ = model(X)
                preds.append(y_pred.cpu().numpy())
            elif model_type in ["SimpleLSTM", "Transformer", "Ensemble"]:
                y_pred = model(X)
                preds.append(y_pred.cpu().numpy())
                loss_function = nn.MSELoss()
                loss = loss_function(y_pred, y) 

            test_loss += loss.item()

            # Optional: Log the loss for each batch
            # wandb.log({f"batch_{step_name}_loss": loss.item(), "epoch": epoch})

    preds = np.concatenate(preds, axis=0)  # Concatenate all predictions
    true_values = np.concatenate(true_values, axis=0)  # Concatenate all true values

    test_loss = test_loss / len(data_loader)
    return test_loss, preds, true_values 

# This is without early stoppign
def train(model, train_loader, valid_loader, test_loader, optimizer, epochs, scheduler, device, 
            model_type:str):

    total_test_loss = 0
    num_epochs = 0

    for epoch in range(epochs):

        train_loss = train_step(model, train_loader, optimizer, device, model_type, clip = 0.3)
        validation_loss, _, _ = test_step(model, valid_loader, device, model_type, step_name="validation")
        test_loss, _, _ = test_step(model, test_loader, device, model_type, step_name="test")

        total_test_loss += test_loss
        num_epochs += 1

        # Log epoch-level metrics
        # wandb.log({"train_loss": train_loss, "validation_loss": validation_loss, "test_loss": test_loss, 
        #            "epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']})


        scheduler.step(validation_loss)

        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss: .6f} | "
            f"Validation Loss: {validation_loss: .6f} | "
            f"Test Loss: {test_loss: .6f} | ",
            f"Learning rate: {optimizer.param_groups[0]['lr']}"
        )
    
    average_test_loss = total_test_loss / num_epochs
    print(f"Average Test Loss: {average_test_loss: .6f}")

    return model

# #This is with early stopping
# def train(model, train_loader, valid_loader, test_loader, optimizer, epochs, scheduler, device, model_type:str, 
#           early_stop_patience=50):
          
#     early_stop_counter = 0
#     best_val_loss = float('inf')
#     total_test_loss = 0

#     for epoch in range(epochs):

#         train_loss = train_step(model, train_loader, optimizer, device, model_type, kld_weight)
#         validation_loss = test_step(model, valid_loader, device, model_type, step_name="validation")
#         test_loss = test_step(model, test_loader, device, model_type, step_name="test")

#         # Log epoch-level metrics
#         wandb.log({"train_loss": train_loss, "validation_loss": validation_loss, "test_loss": test_loss, 
#                    "epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']})

#         total_test_loss += test_loss
#         scheduler.step(validation_loss)

#         print(
#             f"Epoch: {epoch+1} | "
#             f"Train Loss: {train_loss: .6f} | "
#             f"Validation Loss: {validation_loss: .6f} | "
#             f"Test Loss: {test_loss: .6f} | ",
#             f"Learning rate: {optimizer.param_groups[0]['lr']}"
#         )

#         if validation_loss < best_val_loss:
#             best_val_loss = copy.copy(validation_loss)
#             best_model_weights = copy.deepcopy(model.state_dict())
#             early_stop_counter = 0  # Reset the counter if the validation loss improves
#         else:
#             early_stop_counter += 1
        
#         # Check for early stopping condition
#         if early_stop_counter >= early_stop_patience:
#             torch.cuda.empty_cache()
#             print(f"Early stopping at epoch {epoch+1} | Best epoch: {epoch+1-early_stop_counter} | Best Validation Loss: {best_val_loss:.6f}")
#             break

#     average_test_loss = total_test_loss / early_stop_counter
#     print(f"Average Test Loss: {average_test_loss: .6f}")

#     #Load best state
#     model.load_state_dict(best_model_weights)
#     return model
