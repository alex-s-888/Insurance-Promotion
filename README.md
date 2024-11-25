# Insurance Purchase Prediction

The goal of the project is to build and deploy Machine Learning model that predicts if customer will buy insurance policy based on customer's profile. 

Imagine following use case: company is planning promo campaign where it will send some gifts to encourage customers buy policies. 
The budget of campaign is limited so gifts should be sent only to customers that are likely to do the purchase.


## Prerequisites

This is python (ver 3.11) project. The following packages should be present:
- joblib
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn

(Optional) Docker should be installed if you want to deploy resulting model as docker image.    


## Project structure

### dataset
Folder contains [InsurancePolicyHistoricalData.csv](dataset/InsurancePolicyHistoricalData.csv) spreadsheet with customers' data.

### model
Contains the source code  
- [notebook.ipynb](model/notebook.ipynb) - notebook used for EDA and feature selection (may be run in Google Colab).
- [train.py](model/train.py) - python script that builds model and serializes it as `.joblib` file.

### deployment_docker
Folder contains files necessary to deploy model as Docker image.

### test
Instructions how to use/test predictions using dockerized model. 

