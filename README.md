
<div align="center">
<img src="https://github.com/user-attachments/assets/1197a5b8-c82b-4f5c-991c-24b5e5440936" width="700">
</div>

# Machine Learning Integration with Bayesian Algorithms for Passive Microwave Precipitation Retrieval
by Reyhaneh Rahimi and Ardeshir Ebtehaj
## Overview

This repository contains a lightweight implementation of the algorithm presented in the research paper titled "Integration of Machine Learning with a Classic Bayesian Algorithm for Passive Microwave Precipitation Retrievals" for passive microwave precipitation retrievals. The study investigates the integration of machine learning techniques with Bayesian algorithms to enhance the accuracy of precipitation detection and estimation using data from the [Global Precipitation Measurement (GPM)](https://gpm.nasa.gov/) core satellite and the [CloudSat Profiling Radar (CPR)](https://www.jpl.nasa.gov/missions/cloudsat).
## Code Availability
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reyhaneh-92/Sharp_ML/blob/main/Demo_v2.ipynb)


## Abstract

Integration of machine learning with a classic Bayesian algorithm is investigated for passive microwave precipitation retrievals using coincidences from the Global Precipitation Measurement core satellite and the *CloudSat* Profiling Radar (CPR). Among several pixel-level machine learning models, the eXtreme Gradient Boosting Decision Tree (XGBDT), equipped with a weighted cross entropy loss function, exhibits the highest accuracy in detection of precipitation occurrence and phase with a true positive rate greater than 94 (98)% and a false positive rate smaller than 1 (1)% for rainfall (snowfall) over land and oceans with no frozen surfaces. Bayesian retrievals in the feature space of a fully connected multi-layer perception (MLP), equipped with a focal loss function, provide the most accurate estimates of the rates with a mean absolute error of less than 1.50 (0.15) mm/hr for rainfall (snowfall). Mutual information analysis unravels that beyond the near-surface air temperature, the 37 and 183±7(3) GHz are the most informative channels for phase detection over the ocean (land). The physical consistency of the retrievals and new explanations of the precipitation passive microwave signatures are provided through partial dependence analysis and annual comparison with the reanalysis data.

## Features

- **Orbital Retrieval**: Implements a two-step algorithm for retrieving precipitation from observations by the Global Precipitation Measurement Microwave Imager (GMI) and its coincidences with the Cloud Profiling Radar (CPR) onboard CloudSat satellite. 
- **Machine Learning Models**: Implementation of eXtreme Gradient Boosting Decision Tree (XGBDT) combined with a Bayesian passive microwave retrieval algorithm in the embedding space of a Multi-Layer Perceptron (MLP) model.
- **Loss Functions**: Utilization of weighted cross entropy and focal loss functions for improved accuracy in the detection of precipitation occurrence and phase with imbalanced training data.


## DATA and Models

All trained models and the feature dictionaries for Bayesian retrievals are available in a Google Drive folder. Users can download this folder and place it in their own Google Drive or local directory to utilize the provided Google Colab notebook for training and testing with a large dataset. This notebook allows users to run the trained models on a small test set for demonstration and reproducibility of the algorithm.

### Accessing the Data

- **Google Drive Link**: [Download the models and data](https://drive.google.com/drive/folders/1NQFVJy7HsQhYNz35usuZgxmT5HOMeqqy?usp=sharing)
  
### Instructions

1. **Download the Folder**: Access the Google Drive link and download the entire folder containing the models and data.

2. **Upload to Google Drive**: Upload the downloaded folder to your Google Drive account to integrate with the Colab notebook.

3. **Run the Google Colab Notebook**:
   - Open the provided Colab notebook from this repository.
   - Follow the instructions within the notebook to execute the models on the test dataset.
   - Analyze the results and experiment with the models using the test data.

This setup allows users to easily access and run the models without needing extensive local resources, leveraging the power of cloud computing for efficient experimentation.

## Requirements

- Python 3.8+
- XGBoost
- TensorFlow (2.12.0)
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- cartopy (for orbital visualization)





