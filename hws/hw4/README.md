Homework 4: Improving the machine learning pipeline (2.0)
---
Aya Liu 5/30/2019

Running the pipeline
---
```
python3 main.py <data_filepath> <output_directory> <grid_size> <debug>
```
**grid_size** is `test`, `small`, or `large`. It determines the parametergrid for model building.  
**debug** is `True` or `False`. It determines whether print statements are shown.

Analysis
--
`hw4-notebook.ipynb` contains the analysis using the machine learning pipeline for predicting underfunded education projects.

Files
---
- `main.py`: to run in bash
- `pipeline.py`: class for a machine learning pipeline and model performance comparison functions
- `model.py`: class for a machine learning model
- `data_preprocess.py`: dataset-specific functions to 1) clean the dataset before running the pipeline and 2) to preprocess the training and test sets as part of the pipeline
- `data_explore.py`: functions for exploratory data analysis before running the pipeline.  

Dependencies
---
Set up virtual environment
```
python3 -m venv venv
source ./venv/bin/activate
```

Install packages specified in requirements.txt
```
pip -r install requirements.txt
```
