# NER_TokenClassifier
<p align="center"> <img width = "100%" height = "100%" src="imgs/demo.png"/>  </p>

## Introduction

This is finetuning model of BERT to handle NER task from NLP.

## Description

Dataset: `*conll2003*`

Language: `*pytorch*`

# Project Structure

This project follows the following folder structure:

```
ner_root/
├── config/
│   └── ner_config.py
├── data/
│   └── custom_data.py
├── models/
│   ├── preprocessing.py
│   ├── predictor.py
│   ├── ner_model.py
│   └── train.py
└── main.py
```

## Folder Descriptions

### config/
- **ner_config.py**: Contains configuration settings for the project, such as hyperparameters and file paths.

### data/
- **custom_data.py**: Handles data loading, preprocessing, and dataset management.

### models/
- **preprocessing.py**: Contains functions for data preprocessing.
- **predictor.py**: Implements the `Predictor` class for making predictions.
- **ner_model.py**: Defines the model architecture and utilities.
- **train.py**: Contains the training loop and functions to train the model.

### main.py
- The main script to run the project, including loading data, training the model, and making predictions.

## How to use

1. Training:

`python3 train.py`

2. Inference:

`python3 predictor.py`

3. Pipeline for Streamlit Deployment

`streamlit run main.py`
