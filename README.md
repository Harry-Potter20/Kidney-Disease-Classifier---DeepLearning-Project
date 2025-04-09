# Kidney-Disease-Classifier---DeepLearning-Project


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?

### STEPS:

Clone the repo

'''bash
https://github.com/Harry-Potter20/Kidney-Disease-Classifier---DeepLearning-Project
'''
### STEP 01- Create a virtual environement after opening the repo

'''bash
python3 -m venv cnncls
'''

'''bash
source cnncls/bin/activate

### STEP 02- Install the requirements

'''bash
pip install -r requirements.txt
'''

#### cmd
- mlflow ui

#### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Harry-Potter20/Kidney-Disease-Classifier---DeepLearning-Project.mlflow \
MLFLOW_TRACKING_USERNAME=Harry-Potter20 \
MLFLOW_TRACKING_PASSWORD=d094eb8680b50e84269fa4303f872962deca1c15 \
python script.py

Run this to export as env variables

'''bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Harry-Potter20/Kidney-Disease-Classifier---DeepLearning-Project.mlflow

export MLFLOW_TRACKING_USERNAME=Harry-Potter20

export MLFLOW_TRACKING_PASSWORD=d094eb8680b50e84269fa4303f872962deca1c15

'''