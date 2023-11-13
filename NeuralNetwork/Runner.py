from pathlib import Path
import torch
from Framework import TransformerLearning
from DatasetHandler import csvLoader


# Function to load data and create data loaders
def load_data_and_create_loaders(dataset_name):
    # Construct the dataset path using pathlib for better path manipulation
    dataset_path = Path.cwd() / 'Datasets' / dataset_name
    data = csvLoader(dataset_path=dataset_path)

    # Set up data loader parameters
    train_ratio, test_ratio, valid_ratio = 0.7, 0.29, 0.01
    batch_size = 64
    random_split = False
    shuffle = True

    # Create train, test, and validation loaders
    train_loader, test_loader, valid_loader = data.train_test_val_loader(
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        batch_size=batch_size,
        random_splitter=random_split,
        shuffle=shuffle
    )

    return data, train_loader, test_loader, valid_loader


# Function to create the configuration dictionary
def create_configuration(data):
    num_cls = data.num_cls
    # Determine device based on GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        }
    }

    return conf


# Function to perform the learning process
def learning(train_loader, test_loader, valid_loader, conf):
    # Initialize the TransformerLearning class
    transformer_learning = TransformerLearning(train_data=train_loader, test_data=test_loader, conf=conf)

    # Uncomment the lines below if needed for debugging or analysis
    # transformer_learning.check_forward()
    # transformer_learning.check_backward(epochs=100)
    # transformer_learning.select_beast_lr(lrs=[0.9, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001], epochs=10)

    # Run the training process and retrieve loss and accuracy values
    loss_train, loss_test, acc_train, acc_test = transformer_learning.run()

    # Plot the loss and accuracy curves
    transformer_learning.plt_loss(conf['epoch'], loss_train, loss_test)
    transformer_learning.plt_acc(conf['epoch'], acc_train, acc_test)


# Entry point of the script
if __name__ == '__main__':
    # Load data and create loaders
    data, train_loader, test_loader, valid_loader = load_data_and_create_loaders('dataset00.csv')

    # Create configuration dictionary
    conf = create_configuration(data)

    # Perform the learning process
    learning(train_loader, test_loader, valid_loader, conf)
