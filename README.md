# Seismic Phase Classification

The main purpose of this repository is to analyse the deep learning's usage in seismic phase classification. The 
internal dataset (not included in this repo) contains 16 features collected from two seismic stations LPAZ and URZ 
from the last 17 years (?). 

## Datasets

## Scripts

### phase_classification.py
### phase_classification_test.py

## Test Comparison

Following is the accuracy comparison among difference weights against difference test dataset:

|  Weight\Station |  S-LPAZ   |  S-URZ   |  S-ALL   |
| --------------- | ---------:|---------:| --------:|
| W-LPAZ          |   73.12%  |  58.13%  |  65.62%  |
| W-URZ           |   56.96%  |  71.88%  |  64.42%  |
| W-ALL           |   70.00%  |  71.07%  |  70.54%  |