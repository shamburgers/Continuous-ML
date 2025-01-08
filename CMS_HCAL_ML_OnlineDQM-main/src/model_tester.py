
"""
Created on Fri Aug 30 17:05:11 2024

@author: Dale A Julson, DaleAdamJulson@Gmail.com

Adapted from work done by: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: ML Models Test for HCAL DQM Digioccupancy 

"""

import argparse
import os, sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
data_path = os.path.abspath(os.path.dirname(current_path)+"/data")
sys.path.append(data_path)
result_path = os.path.abspath(os.path.dirname(current_path)+"/results")

import utilities as util
from model_datasets import * 
from models_spatial import * 

model_path_template = "{}/{}/model/{}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ad_model_validation(**kwargs):

    default_setting = {
        "dataset": None,
        "model_alg": None,
        "test_dataset": "",
        "trained_model": "",
        "latent_dim": 4, # [2, 4, 8, 16, 32] increase this to improve accuracy and reduce to speed up training
        "rnn_model": "lstm",
        "shuffle": False,
        "num_layers": 1,
        "batch_size": 1
    }
    _kwargs = copy.deepcopy(kwargs)
    kwargs.update(default_setting)
    kwargs.update(_kwargs)

    test_data_filename = kwargs.get("test_dataset", "test_dataset")
    trained_model_filename = kwargs.get("trained_model", "trained_model")
    kwargs_model = json.loads(kwargs.pop("kwargs_model", "{}"))

    print(f"kwargs_model: {kwargs_model}")
    
    kwargs["test_data_filename"] = test_data_filename
    
    model_alg = kwargs.get("model_alg", "CNNRNNAE_MultiDim_SPATIAL")
    shuffle = kwargs.get("shuffle", False)
    batch_size = kwargs.get("batch_size", 4)
    latent_dim = kwargs.get("latent_dim", 32)

    # load test_dataset
    print("loading test_data: {}...".format(test_data_filename))
    test_data = util.load_pickle("{}".format(test_data_filename))
    test_data.memorysize = test_data.timewindow_size
    print("feature_dim: {} target_dim: {}".format(
        test_data.shape(), test_data.shape(target=True)))
    test_data_config = test_data.__dict__.copy()
    test_data_config.pop('samples', None)

    print(f"Test data config: {test_data_config}")

    feature_dim = test_data.shape()[-1]
    target_dim = test_data.shape(target=True)[-1]
    spatial_dims = test_data.shape()[2:-1]
    print(f"spatial_dims: {spatial_dims}")
    if spatial_dims[-1] == 1: spatial_dims = spatial_dims[:-1]
        
    print(f"spatial_dims: {spatial_dims}")

    memorysize = test_data.memorysize
   
    args = kwargs.copy()
    print(f"args: {[args.pop(k, None) for k in ["datatype", "latent_dim", "num_layers"]]}")

    args.update(kwargs_model)

    # Load data into DataLoader
    try:
        dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
        print(f"{'*' * 50}\n" + "Data successfully loaded".center(50) + f"\n{'*' * 50}")
        print(f"Dataset length: {len(dataloader.dataset)}")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Number of batches: {len(dataloader)}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Load pretrained model
    MODEL_SEL = eval(model_alg)
    ae_model = MODEL_SEL(feature_dim, latent_dim, target_dim, spatial_dims,
                            memorysize=memorysize,
                            **args
                            )
    print("loading model: {}...".format(trained_model_filename))

    trained_model_file = util.load_pickle(trained_model_filename)
    ae_model = trained_model_file["model"]
    print(f"Loaded model type: {type(ae_model)}")

    ae_model.eval()
    ae_model.criterion = nn.MSELoss()

    reconstruction_error_list = []

    with torch.no_grad():

        for test_data in dataloader:

            inputs, targets = test_data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = ae_model(inputs)
            # calculate loss
            if isinstance(outputs, tuple):
                loss = ae_model.loss(targets, *outputs, **kwargs)
            else:
                loss = ae_model.loss(targets, outputs, **kwargs)
            reconstruction_error_list.append(loss)

    print(f"{'*' * 50}\n" + f"Average MSE is: {(sum(reconstruction_error_list)/len(reconstruction_error_list)):.2e}".center(50) + f"\n{'*' * 50}")

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_error_list, label='Test data')
    # plt.xlabel('Batch')
    plt.ylabel('Reconstruction Error (MSE)')
    # plt.title('Model Testing')
    plt.legend()
    plt.grid(True)
    # plt.savefig(result_path+'Testing_MSE.pdf')
    plt.show()

if __name__ == '__main__':
    # constructing argument parsers
    parser = argparse.ArgumentParser(description="gpu model trainer")
    parser.add_argument('-ma', '--model_alg', default="",
                        help='model algoritm selection')
    parser.add_argument('-td', '--test_dataset', type=str,
                        default="data set to perform testing on (including full directory path)",
                        help='data source selection')
    parser.add_argument('-tm', '--trained_model',
                        type=str,
                        help='previously trained model in dataset filename (including full directory path)')
    parser.add_argument('-k', '--kwargs', type=str, default="",
                        help='training parameters')
    parser.add_argument('-km', '--kwargs_model', type=str, default="{}",
                        help='kwargs_model such as dict \\"use_attention: false, nodropout: [encoder]\\"')


    args = parser.parse_args()
    print(args)
    args = vars(args)
    ad_model_validation(**args)

# python3 model_tester.py -ma CNNFNNAE_MultiDim_SPATIAL -td data/HE/HE_ts1_test_dataset_2018_iid.pkl -tm results/data_HE_2018_FNN_ts1/model/he_m_1_is_minmax_ns_1/CNNFNNAE_MultiDim_SPATIAL_3582/AE_MODEL.pkl
