# CMS_HCAL_ML_OnlineDQM (Workgroup resource sharing repository)
 Machine learning for online-DQM-HCAL monitoring 

 ## Root TH3 file (generated using HcalAnalyzer) to numpy conversion for the ML HCAL DQM Digioccupancy  
 
Convertor script is in ![hcalanalyzer_root_to_numpy.py](./src/hcalanalyzer_root_to_numpy.py) 

Example:
 
    # set file path
    filepath = r"..\data\HBHE\Run355456__hehb_cut20fc__depth.root" # change path to your local path
    
    # with plots
    TH3Obj_np, run_ls_lumi_df, meta_data = rootfile_loader(filepath, depth_size=7, isrun_setting=True, isplot=True) # change depth_size according to the included number of hist3d_depthX histograms in the root file

    # or without plots, faster processing
    TH3Obj_np, run_ls_lumi_df, meta_data = rootfile_loader(filepath, depth_size=7, isrun_setting=True, isplot=False)

### Conversion results: 

![Extracted digi-occupancy histogram maps per depth](./results/screenshot_Run355456__hehb_cut20fc__depth.png)
![Extracted number of events per lumisection](./results/NumEvents_Run355456__hehb_cut20fc__depth.png)

## Subdetector Segmentation Map Generation from HCAL-Emap 
Generates the spatial mask [64 x 72 x 7] to select the active channels per a given subdetector w.r.t to digi-occupancy map

Notebook: prepare_hcal_seg_subdetector_mask
![prepare_hcal_seg_subdetector_mask notebook](./examples/prepare_hcal_seg_subdetector_mask.ipynb)

Script: 

    # Set first the desired target subdetector in the prepare_hcal_seg_subdetector_mask.py and run the script
    python src/prepare_hcal_seg_subdetector_mask.py
    
## Dataset Preparation and Exploration

Select the target subdetector name: ["he", "hb", "hf", "ho"]

Master dataset preparation: Exports list of runs from root to NumPy. This creates "master_dataset" directory. 

Exploration dataset preparation: Prepare the dataset for data exploration and analysis. This creates "explore_dataset" directory. 

Train and test dataset preparation: Prepare data sets for later to be employed for ML modeling. This creates "train_dataset" and "tes_dataset" directories. 

Data sets are stored in "/data/(main_rootfiles_dataset_dirname)/"

Notebook: prepare_and_explore_datasets
![prepare_and_explore_datasets notebook](./examples/prepare_and_explore_datasets.ipynb)

Script: 

    # set first the desired target subdetector in the prepare_and_explore_datasets.py and run the script
    python src/prepare_and_explore_datasets.py
    
## AD Modeling Development

Generates modeling data sets (from the previously prepared training and testing Numpy data sets) as objects of the torch.Datasets.  
The data sets are stored in Pickle format for model training and evaluation (![model_datasets.py](./src/model_datasets.py).

The modeling data sets models are stored in "/data/"

AD Models are designed in![models_spatial.py](./src/models_spatial.py) 

Some of the supported models are:

    1) Non-temporal (CNN+AE, CNN+VAE, ResCNN+AE. ResCNN+VAE) and 
    2) Temporal (CNN+RNN+AE, CNN+RNN+VAE, ResCNN+RNN+AE, ResCNN+RNN+VAE, the RNN: {RNN, GRU or LSTM})

Model training scripts are given in ![model_trainer.py](./src/model_trainer.py)

Trained models are stored in "/results/(unique_dataset_name)/(model_config_based_nametag)/(unique_model_name)"

Script (Non-temporal Modeling Data sets): 

    # TRAIN
    python model_datasets.py -it -s he -d ts -m 1 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_train_dataset" -td he_train_dataset_iid 
    
    # TEST
    python model_datasets.py -s he -d ts -m 1 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_test_dataset" -td he_test_dataset_iid -tr he_train_dataset_iid


Script (Temporal Modeling Data sets): 

    # TRAIN
    python model_datasets.py -it -s he -d ts -m 5 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_train_dataset" -td he_train_dataset_ts 
    
    # TEST
    python model_datasets.py -s he -d ts -m 5 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_test_dataset" -td he_test_dataset_ts -tr he_train_dataset_ts

Script (Non-temporal AD Model Training):    

    python model_trainer.py -ma CNNFNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_iid

Script (Temporal AD Model Training):    

     python model_trainer.py -ma CNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts


Script (Temporal + Graph AD Model Training):
      
     python model_trainer.py -ma CNNGNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"node_edge_index_filename\":\"C:\\Users\\mulugetawa\\OneDrive - Universitetet i Agder\\CERN\\InductionProject\\CMS_HCAL_ML_OnlineDQM\\data\\HCAL_CONFIG\\he_graph_edge_array_indices_depth.npy\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts 


Script (Temporal + Graph AD + GraphSTAD_Optimizer (RIN) Model Training):

     python model_trainer.py -ma CNNGNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"use_graphstad_optimizer\":\"true\", \"node_edge_index_filename\":\"C:\\Users\\mulugetawa\\OneDrive - Universitetet i Agder\\CERN\\InductionProject\\CMS_HCAL_ML_OnlineDQM\\data\\HCAL_CONFIG\\he_graph_edge_array_indices_depth.npy\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts


Script (Temporal + GraphSTAD_Optimizer (RIN) AD Model Training):
     
    python model_trainer.py -ma CNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"use_graphstad_optimizer\":\"true\", \"subdetector_name\":\"he\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts


Script (Temporal + GraphSTAD_Optimizer (RIN + DC) AD Model Training): 

    python model_trainer.py -ma CNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"use_graphstad_optimizer\":\"true\", \"subdetector_name\":\"he\", \"use_spatial_split\":\"true\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts



Notebook: ad_model_developement notebook
![ad_model_developement notebook](./examples/ad_model_developement.ipynb)


## Sample AD Results

### HCAL Endcal: spatial hot channel anomaly, synthetically generated
![HCAL Endcal: spatial hot channel anomaly, synthetically generated](./results/he_digi_spatial_flag_with_channel_anomaly_synthetic.png)

### HCAL Endcal: spatial digi-occupancy map with injected hot channel anomaly
![HCAL Endcal: spatial digioccupancy map with injected hot channel anomaly](./results/he_digi_spatial_map_with_channel_anomaly_synthetic.png)

### HCAL Endcal: reconstructed spatial digi-occupancy map
![HCAL Endcal: reconstructed spatial digi-occupancy map](./results/he_digi_spatial_map_with_channel_anomaly_synthetic_pred.png)

### HCAL Endcal: spatial map of the anomaly scores
![HCAL Endcal: spatial map of the anomaly scores](./results/he_digi_spatial_map_with_channel_anomaly_synthetic_score.png)

