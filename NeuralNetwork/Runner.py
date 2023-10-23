from Framework import TransformerLearning
from DatasetHandler import csvLoader
from pathlib import Path
import torch


def load_data_and_create_loaders(dataset_name):
    dataset_path = f'{Path.cwd()}/Datasets/{dataset_name}'
    data = csvLoader(dataset_path=dataset_path)
    # data.feature_engineering(normalize=True, features_prune=True, dimension_reduction=True, nan_control=True, text_encode=True)
    train_ratio, test_ratio, valid_ratio = 0.7, 0.29, 0.01
    batch_size = 64
    random_split = False
    shuffle = True
    train_loader, test_loader, valid_loader = data.train_test_val_loader(
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        batch_size=batch_size,
        random_splitter=random_split,
        shuffle=shuffle
    )
    return data, train_loader, test_loader, valid_loader


def create_configuration(data):
    num_cls = data.num_cls
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        }
    }
    return conf

def learning(train_loader, test_loader, valid_loader, conf):
    transformer_learning = TransformerLearning(train_data=train_loader, test_data=test_loader, conf=conf)
    # transformer_learning.check_forward()
    # transformer_learning.check_backward(epochs=100)
    # transformer_learning.select_beast_lr(lrs=[0.9, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001], epochs=10)
    loss_train, loss_test, acc_train, acc_test = transformer_learning.run()

    transformer_learning.plt_loss(conf['epoch'], loss_train, loss_test)
    transformer_learning.plt_acc(conf['epoch'], acc_train, acc_test)


if __name__ == '__main__':
    data, train_loader, test_loader, valid_loader = load_data_and_create_loaders('test0.csv')
    conf = create_configuration(data)
    learning(train_loader, test_loader, valid_loader, conf)


