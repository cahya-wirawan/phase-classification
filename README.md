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
with Tensorflow as backend. Keras is used intead of Tensorflow directly to simplify the prototyping. The training will 
generate a weight and model files: phase_weights_best_&lt;station name&gt;.hdf5 and 
phase_model_best_&lt;station name&gt;.hdf5 in "results" directory which will be used later for testing purpose.

### Usage

* List of all possible commands:
```
$ python phase_classification.py -h                                                                                      
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

* Training (the default model uses 11 hidden layers "128 128 64 48 48 32 32 48 32 16" and 2000 epochs):
```
$ python phase_classification.py
Using TensorFlow backend.
N: 10000 entries
P: 6000 entries
S: 3000 entries
T: 8000 entries
Summary: 27000 entries
/home/cahya/.virtualenvs/phase/lib/python3.5/site-packages/sklearn/base.py:115: DeprecationWarning: Estimator KerasClassifier modifies parameters in __init__. This behavior is deprecated as of 0.18 and support for this behavior will be removed in 0.20.
  % type(estimator).__name__, DeprecationWarning)
2018-02-01 12:01:20.894296: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-02-01 12:01:21.017779: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:895] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-02-01 12:01:21.019589: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1105] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8855
pciBusID: 0000:28:00.0
totalMemory: 7.92GiB freeMemory: 7.34GiB
2018-02-01 12:01:21.019611: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:28:00.0, compute capability: 6.1)
...
Baseline: 70.54% (1.90%)

```
* Training using only 3 hidden layers "4 4 4", dopout of 0.3 and 10 epochs:
```
$ python phase_classification.py -l "4 4 4" -d 0.3 -e 10
...
Baseline: 37.60% (4.20%)
```
* Testing :
```
$ python phase_classification.py -a test
Using TensorFlow backend.
N: 1000 entries
P: 600 entries
S: 240 entries
T: 400 entries
Summary: 2240 entries
2018-02-06 12:16:26.875535: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-02-06 12:16:26.998599: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:895] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-02-06 12:16:26.998962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1105] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8855
pciBusID: 0000:28:00.0
totalMemory: 7.92GiB freeMemory: 7.34GiB
2018-02-06 12:16:26.998977: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:28:00.0, compute capability: 6.1)
Loaded model from disk
acc: 70.54%
Confusion matrix:
              P     S     T     N 
        P 413.0   1.0 126.0  60.0 
        S   9.0 121.0  33.0  77.0 
        T  73.0  39.0 228.0  60.0 
        N  47.0  67.0  68.0 818.0 
```
## Test Comparison

Following is the accuracy comparison among difference weights against difference test dataset:

|  Weight\Station |  S-LPAZ   |  S-URZ   |  S-ALL   |
| --------------- | ---------:|---------:| --------:|
| W-LPAZ          |   73.12%  |  58.13%  |  65.62%  |
| W-URZ           |   56.96%  |  71.88%  |  64.42%  |
| W-ALL           |   70.00%  |  71.07%  |  70.54%  |