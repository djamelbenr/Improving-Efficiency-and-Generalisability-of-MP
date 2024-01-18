# Improving-Efficiency-and-Generalisability-of-Motion-Predictions-with-Deep-Multi-Agent-Learning-and-Multi-Head-Attention

![visitors](https://visitor-badge.glitch.me/badge?page_id=djamelbenr/Motion-prediction-with-Deep-Multi-Agent-Learning-and-Multi-Head-Attention)

This work aims to design a data-driven prediction framework for highly Automated Vehicles (AVs) that utilises multiple inputs to prove a multimodal, probabilistic estimate of the future intentions and trajectories of surrounding vehicles in freeway operation. Our proposed framework is a deep multi-
agent learning-based system designed to effectively capture social interactions (please refer to our recent [paper](https://doi.org/10.1109/TITS.2022.3207347) between vehicles without relying on map information. This algorithm achieved a good prediction performance with a lower prediction error in real traffic data at highways. Evaluation of the proposed framework using the NGSIM (US-101 and I-80) and HighD datasets shows satisfactory prediction performance for long- term trajectory prediction of multiple surrounding vehicles. Additionally, the proposed framework has higher prediction accuracy and generalisability than state-of-the-art approaches.

[Djamel Eddine Benrachou](https://scholar.google.com/citations?user=0lHe7ycAAAAJ&hl=en)

[Centre for Accident Research & Road Safety - Queensland, QUT](https://research.qut.edu.au/carrsq/staff/djamel-benrachou/)

------------------

**News 🚀🚀**
- [04/2023] Ongoing repo update. The most recent work...
- [03/2023] Work on Transactions on Intelligent Transportation Systems (T-ITS, Q1-JCR)...
- [09/2022] Work on Transactions on Intelligent Transportation Systems (T-ITS, Q1-JCR): 
            "Use of Social Interaction and Intention to Improve Motion Prediction Within Automated Vehicle Framework: A Review"
            
------------------

This is the official repository and [PyTorch](https://pytorch.org/) implementations of different works presented at:
To be added ..

Our papers:
- [Improving Efficiency and Generalisability of Motion Predictions with Multi-Agent Learning and Multi-Head Attention](https://arxiv.org/) in IEEE Transactions on Intelligent Transportation Systems
- [Use of Social Interaction and Intention to Improve Motion Prediction Within Automated Vehicle Framework: A Review](https://ieeexplore.ieee.org/abstract/document/9904961) in IEEE Transactions on Intelligent Transportation Systems


# Setup：
The code can be executed in a conda environment--please follow the tutorial below
```
conda create -n MyEnvi python=3.7
source activate MyEnvi

conda install pytorch==1.12.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install tensorboard=1.14.0
conda install numpy=1.16 scipy=1.4 h5py=2.10 future
```
The code can also be written in the following environment:
```
python 3.7.11
pytorch 1.10.0
cuda 11.3 or above
cudnn 8.2.0
```
If available, check `requirements.txt`

```
conda create --name GATLSTM_env python=3.7 \
conda install -n GATLSTM_env ipykernel --update-deps --force-reinstall
```
```
python3 -m pip install --upgrade pip \
python3 -m pip install --upgrade Pillow \
```
```
pip install \
    prodict \
    torch \
    pyyaml \
    torchvision \
    tensorboard \
    glob2 \
    matplotlib \
    sklearn \
    gitpython \
    thop \
    fvcore \
    torchstat \
    torchsummary \
    ipykernel \
    sns
```
In order to install torch-geometric and associated modules for the GNN computations, please check the following documentation:

https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

E.g.

```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```

# Datasets

## NGSIM
  
From NGSIM website:  
* Register at [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)  
* Download [NGSIM-US-101-LosAngeles-CA]US-101-LosAngeles-CA.zip and [NGSIM-I-80-Emeryville-CA] I-80-Emeryville-CA.zip  
* Unzip and extract vehicle-trajectory-data into raw/us-101 and ./raw/i-80  
  
From googledrive:  
* Download i-80: https://drive.google.com/open?id=19ovxiJLCnS1ar1dYvUIrfdz_UnGejK-k  
* Download us-101: https://drive.google.com/open?id=14dMKew22_5evfOoSGEBYqHP3L92e7nyJ  

Dataset fields:  
* doc/trajectory-data-dictionary.htm  

This Dataset is to be pre-processed with the Matlab function preprocess_data.m (that is a slightly modified version of the one from https://github.com/nachiket92/conv-social-pooling)

* Already preprocessed samples: (to run tests on Hybrid-Models)
    * https://drive.google.com/open?id=171YZvK2DnJbtIgve6qAOkk6AFNrGAYTJ
    * https://drive.google.com/open?id=1N080iiQc43MTvLMqaLtsZh49okJL0hPx
    
Dataset fields:  
* doc/trajectory-data-dictionary.htm  

## HighD 
   * [https://www.highd.com/](https://www.highd-dataset.com/)

## Pre-processed NGSIM and HighD can be downloaded from [here](https://connectqutedu-my.sharepoint.com/:u:/r/personal/n10439251_qut_edu_au/Documents/processed_data/OneDrive_1_4-20-2023.zip?csf=1&web=1&e=uRNYbS)
## Argoverse

Get the data from the Argoverse website:
* https://www.argoverse.org/

Put their API from Argoverse in a directory at same depth named "Argoverse"
* https://github.com/argoai/argoverse-api

# Special instructions for Hybrid Models: 
Setup the name of the trained model in the load_name field of config file (.yaml) (follow the example: <model>_<dataset>_<id>)
Run python ```test_models/save_trained_model.py``` to save the prediction results computed on the test set (after training).
Run python ```test_models/statistic_result.py``` to print metrics evaluation (RMSE,NLL, etc.), plot covariance matching and error histogram (from the saved results).




Data Preparation:
The raw data of Next Generation Simulation (NGSIM) is downloadable at https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

1. Download the data:      Once the data downloaded the raw data into ```./raw```
2. Preprocess the data : Run 
```
preprocess_data.m to pre-preprocess the raw data (note: the pre-processed data will be uploaded on cloud).
```
This repository is for testing the trained models for motion prediction on highways. No end-to-end trainer is needed here.
3.To use the pretrained models at ./ .py and evaluate the models performance run! 🎉 🎉
```
python3 evaluate.py --name ngsim_model --batch_size 64 \
    --test_set ./datasets/NGSIM/test.mat 
```   

Trained models can be downloaded [here](https://connectqutedu-my.sharepoint.com/)


  
 
# Citation

@article{benrachou2023improving,
  title={Improving Efficiency and Generalisability of Motion Predictions With Deep Multi-Agent Learning and Multi-Head Attention},
  author={Benrachou, Djamel Eddine and Glaser, Sebastien and Elhenawy, Mohammed and Rakotonirainy, Andry},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}

 # License
