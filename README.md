# Seismic Phase Classification

This project is an attempt to apply deep learning on seismic phase classification. 

## The Datasets
The project uses an internal dataset (not included in this repo) collected from two seismic stations LPAZ 
and URZ from the last several years (?). It contains 776155 entries and 16 features. The dataset  
ml_feature_bck2.csv is a dump of database table ml_feature_bck2, which is created/collected by Radek Hofman.
Furthermore, it is splited using the python script phase_splitter.py into two files (ml_feature_bck2_train.csv 
and ml_feature_bck2_test.csv) for training (and validation) dataset and test dataset. The test dataset 
contains following entries

|  Station and Phase | #Entries    |
| ------------------ |  ----------:|
|  LPAZ P-Phase      |   300       |
|  LPAZ S-Phase      |   120       |
|  LPAZ T-Phase      |   200       |
|  LPAZ Noise        |   500       |
|  URZ P-Phase       |   300       |
|  URZ S-Phase       |   120       |
|  URZ T-Phase       |   200       |
|  URZ Noise         |   500       |


## Datasets Visualization
t-Distributed Stochastic Neighbor Embedding (t-SNE) is used to reduce the dimensionality of the dataset
and visualize it in 2D graph. 
![16D Phases in 2D](https://github.com/cahya-wirawan/phase-classification/blob/master/images/4Phases-tSNE.jpg)

## The Application

### phase_classification.py

## Test Comparison

Following is the accuracy comparison among difference weights against difference test dataset:

|  Weight\Station |  S-LPAZ   |  S-URZ   |  S-ALL   |
| --------------- | ---------:|---------:| --------:|
| W-LPAZ          |   73.12%  |  58.13%  |  65.62%  |
| W-URZ           |   56.96%  |  71.88%  |  64.42%  |
| W-ALL           |   70.00%  |  71.07%  |  70.54%  |