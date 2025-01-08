# CML_Project_with_Cerium_Labs

## Overview

Repository for Continuous Machine Learning project. As of the moment, the following cases were handled using four models, namely `LSTMVAE`, `SimpleLSTM`, `Transformer`, and `Ensemble`. 

Only on a subset of the entire dataset is considered, as theory demands it this way. That is, model training and inference is only dedicated to a particular element so that input shape goes as [cps, 1], where 1 serves as the element of interest.

A `create_datasets.ipynb` can be found under the folder `src`. 
Other relevant source codes that can be tweaked are also located in `src`. There are also available documentations within each python file.


### 

## Dependencies

- [Python >= 3.8](https://packaging.python.org/en/latest/tutorials/installing-packages/)
- [Pytorch](https://pytorch.org/)
- [wandb](https://docs.wandb.ai/quickstart)