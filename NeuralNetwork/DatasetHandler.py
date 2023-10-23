from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data.dataset import Dataset
from FeatureEngineering import Preprocessor
import torch
import pandas as pd
import numpy as np
import glob
import math


class csvLoader(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.df = pd.read_csv(dataset_path)
        # self.df = pd.read_pickle(dataset_path)


    def feature_engineering (self, normalize, features_prune, dimension_reduction, nan_control, text_encode):

        self.df = Preprocessor(self.df, normalize, features_prune, dimension_reduction, nan_control, text_encode )


    def train_test_val_loader(self, train_ratio, test_ratio, valid_ratio, random_splitter, batch_size, shuffle):
        cls_idx = len(self.df.columns) - 1
        data = self.df.values
        x_data = data[:, :cls_idx]
        y_data = data[:, cls_idx]

        x = torch.FloatTensor(x_data.astype(np.float64))
        y = torch.LongTensor(y_data.astype(np.int8))

        dataset =  TensorDataset(x, y)

        dataset_size = len(dataset)
        train_size = math.floor(train_ratio * dataset_size)
        test_size = math.floor(test_ratio * dataset_size)
        valid_size = math.floor(valid_ratio * dataset_size)
        valid_size += (len(dataset)) - (train_size + test_size + valid_size)

        if random_splitter:
            train_dataset, test_dataset, valid_dataset = random_split(dataset, [train_size, test_size, valid_size])
        else:
            train_dataset = TensorDataset(*dataset[:train_size])
            test_dataset = TensorDataset(*dataset[train_size:(train_size+test_size)])
            valid_dataset = TensorDataset(*dataset[(train_size+test_size):])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

        self.num_cls = len(y.unique())

        return train_loader, test_loader, valid_loader


