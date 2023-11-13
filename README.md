<!-- Project Title -->
# Transformer-Based Machine Learning Framework

<!-- Project Description -->
A comprehensive machine learning framework for building and training transformer-based models using Python and PyTorch.

<!-- Table of Contents -->
## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#Dataset)
- [Run](#Run)

<!-- Introduction -->
## Introduction

![Transformer Image](images/transformer.png)

This project is designed to simplify the development and training of transformer-based models for various tasks, such as text classification, language modeling, and more. It provides data loading, preprocessing, model configuration, training, and evaluation components.

**Key Features**:
- Data loading and preprocessing
- Customizable training parameters
- Model training and evaluation
- GPU acceleration support

<!-- Prerequisites -->
## Prerequisites

Before using this framework, ensure you have the following prerequisites installed on your system:

- **Python**: Version 3.10
- **PyTorch**: Version 2.1.0

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

<!-- Installation -->
## Installation

To get started with the Transformer-Based Machine Learning Framework, follow these steps:

Clone the repository to your local machine:

```bash
https://github.com/ms-unlimit/Transformer-Based-Machine-Learning-Framework.git
```

<!-- Dataset -->
## Dataset

The dataset used in this project is expected to be in CSV format, where the last column represents the labels, and the preceding columns represent features.

### CSV Format

The CSV file should adhere to the following format:

```plaintext
feature_1, feature_2, ..., feature_n, label
value_11, value_12, ..., value_1n, label_1
value_21, value_22, ..., value_2n, label_2
...
value_m1, value_m2, ..., value_mn, label_m
```

<!-- Run -->
## Run

To run the Transformer-Based Machine Learning Framework, follow these steps:

### 1. Configuration in Runner.py

Open the `Runner.py` file and set the dataset and model configurations. Example configurations are provided below:

```python
    
    # Configuration for Dataset
    data, train_loader, test_loader, valid_loader = load_data_and_create_loaders('dataset_name.csv')
    
    train_ratio, test_ratio, valid_ratio = 0.7, 0.29, 0.01
    batch_size = 64
    random_split = False
    shuffle = True

    # Configuration dictionary for training
    conf = {
        'device': device,
        'epoch': 200,
        'optimizer': {'lr': 0.01, 'momentum': 0.9},
        'accuracy': {'task': 'multiclass', 'num_classes': num_cls},
        'model_param': {
            'd_model': 1024,
            'nhead': 32,
            'num_enc': 32,
            'd_feed': 2048,
            'dropout': 0.2,
            'activation': 'gelu',
            'num_cls': num_cls,
```
### 1. Run the Script
python Runner.py



