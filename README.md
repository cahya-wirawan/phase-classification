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
and visualize it in 2D graph [Jupyter notebook's phase_tSNE](https://github.com/cahya-wirawan/phase-classification/blob/master/phase_tsne.ipynb) . 
![16D Phases in 2D](https://github.com/cahya-wirawan/phase-classification/blob/master/images/4Phases-tSNE.jpg)

## The Application

### phase_classification.py
This is the main application for the training and testing. The deep learning model is implemented using Keras 
with Tensorflow as backend. Keras is used intead of Tensorflow directly to simplify the prototyping.

### Usage
```
$ python phase_classification.py -h                                                               
/home/cahya/.virtualenvs/phase/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.                                                    
  from ._conv import register_converters as _register_converters                                                                                           
Using TensorFlow backend.                                                                                                                                  
usage: phase_classification.py [-h] [-a {train,test}]                                                                                                      
                               [--train_dataset TRAIN_DATASET]                                                                                             
                               [--test_dataset TEST_DATASET] [-e EPOCHS]                                                                                   
                               [-l LAYERS] [-d DROPOUT] [-s STATION]                                                                                       
                               [-v VERBOSE] [-P P] [-S S] [-T T] [-N N]                                                                                    
                                                                                                                                                           
optional arguments:                                                                                                                                        
  -h, --help            show this help message and exit                                                                                                    
  -a {train,test}, --action {train,test}                                                                                                                   
                        set the action, either training or test the dataset                                                                                
                        (default: train)                                                                                                                   
  --train_dataset TRAIN_DATASET                                                                                                                            
                        set the path to the training dataset (default:                                                                                     
                        data/phase/ml_feature_bck2_train.csv)                                                                                              
  --test_dataset TEST_DATASET                                                                                                                              
                        set the path to the test dataset (default:                                                                                         
                        data/phase/ml_feature_bck2_test.csv)                                                                                               
  -e EPOCHS, --epochs EPOCHS                                                                                                                               
                        set the epochs number) (default: 2000)                                                                                             
  -l LAYERS, --layers LAYERS                                                                                                                               
                        set the hidden layers) (default: 128 128 64 48 48 32                                                                               
                        32 48 32 16)                                                                                                                       
  -d DROPOUT, --dropout DROPOUT                                                                                                                            
                        set the dropout) (default: 0.1)                                                                                                    
  -s STATION, --station STATION                                                                                                                            
                        set the station name, it supports currently only LPAZ,                                                                             
                        URZ and ALL (default: ALL)                                                                                                         
  -v VERBOSE, --verbose VERBOSE                                                                                                                            
                        set the verbosity) (default: 0)                                                                                                    
  -P P                  set the number of entries of P to be read from the                                                                                 
                        dataset) (default: 6000)                                                                                                           
  -S S                  set the number of entries of S to be read from the                                                                                 
                        dataset) (default: 3000)                                                                                                           
  -T T                  set the number of entries of T to be read from the                                                                                 
                        dataset) (default: 8000)                                                                                                           
  -N N                  set the number of entries of N to be read from the                                                                                 
                        dataset) (default: 10000)                
```

## Test Comparison

Following is the accuracy comparison among difference weights against difference test dataset:

|  Weight\Station |  S-LPAZ   |  S-URZ   |  S-ALL   |
| --------------- | ---------:|---------:| --------:|
| W-LPAZ          |   73.12%  |  58.13%  |  65.62%  |
| W-URZ           |   56.96%  |  71.88%  |  64.42%  |
| W-ALL           |   70.00%  |  71.07%  |  70.54%  |