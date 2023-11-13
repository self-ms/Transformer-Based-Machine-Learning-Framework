import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Models import TransformerModel
from MetricCalculator import AverageMeter
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from statistics import mean
import logging

logging.basicConfig(level=logging.INFO)

class Framework:
    def __init__(self, mydataset=None, train_dataset=None, valid_dataset=None, test_dataset=None, conf=None, test_dataset_other=None):
        """
        Initialize the training framework.

        Args:
            mydataset: Main dataset
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            test_dataset: Test dataset
            conf: Configuration parameters
            test_dataset_other: Another test dataset
        """
        self.mydataset = mydataset
        self.test_dataset_other = test_dataset_other
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.conf = conf
        logging.info("Initializing Framework")

    def run(self, my_model, optimizer, loss_function):
        """
        Train and evaluate the model.

        Args:
            my_model: The neural network model
            optimizer: The optimizer for training
            loss_function: The loss function
        """
        start_epoch_number = 0

        try:
            logging.info("Loading checkpoint ...")
            loaded_checkpoint = torch.load("checkpoint.pth")
            my_model.load_state_dict(loaded_checkpoint["model_state"])
            optimizer.load_state_dict(loaded_checkpoint["optim_state"])
            start_epoch_number = loaded_checkpoint["epoch"]
            logging.info(f'Continuing from epoch [{start_epoch_number}]')
        except Exception as e:
            logging.warning(f"Failed loading: {e}")

        dataloader = DataLoader(self.train_dataset, batch_size=self.conf.batch_size, shuffle=True)

        for epoch in range(start_epoch_number, self.conf.number_of_epoch):
            loss = self.train(my_model, optimizer, loss_function, dataloader)

            if epoch % 5 == 0:
                self.validation(my_model, loss_function, loss, epoch)

        self.test(my_model)

    def train(self, my_model, optimizer, loss_function, dataloader):
        """
        Train the model.

        Args:
            my_model: The neural network model
            optimizer: The optimizer for training
            loss_function: The loss function
            dataloader: Data loader for training data

        Returns:
            loss: The training loss
        """
        for data_batch, label_batch in dataloader:
            optimizer.zero_grad()
            out = my_model(data_batch)
            loss = loss_function(out, label_batch)
            loss.backward()
            optimizer.step()
        return loss

    def validation(self, my_model, loss_function, loss, epoch):
        """
        Perform validation and print results.

        Args:
            my_model: The neural network model
            loss_function: The loss function
            loss: Training loss
            epoch: Current epoch number
        """
        my_model = my_model.eval()
        out = my_model(self.mydataset.x[self.valid_dataset.indices, :self.conf.number_of_features])
        _, predicted = torch.max(out.data, 1)
        label_v = self.mydataset.y[self.valid_dataset.indices]
        loss_v = loss_function(out, label_v)
        logging.info(f'Epoch [{epoch + 1}/{self.conf.number_of_epoch}] Train Loss: {loss:.4f}')
        logging.info(f'Epoch [{epoch + 1}/{self.conf.number_of_epoch}] Valid Loss: {loss_v:.4f}')
        acc = (100 * torch.sum(label_v == predicted) / self.number_of_valid)
        logging.info(f'Accuracy of the network in Validation: {acc:.4f}%')

    def test(self, my_model):
        """
        Test the model on test datasets and print results.

        Args:
            my_model: The neural network model
        """
        my_model = my_model.eval()
        X = self.mydataset.x[self.test_dataset.indices]
        Y = self.mydataset.y[self.test_dataset.indices]
        out = my_model(X)
        _, predicted = torch.max( out.data, 1)
        test_accuracy = 100 * torch.sum(Y == predicted) / self.number_of_test
        logging.info(f'Accuracy of the network in Test: {test_accuracy:.4f}%')

        X2 = self.test_dataset_other.x
        Y2 = self.test_dataset_other.y
        out2 = my_model(X2)
        _, predicted2 = torch.max(out2.data, 1)
        other_test_accuracy = 100 * torch.sum(Y2 == predicted2) / len(self.test_dataset_other)
        logging.info(f'Accuracy of the network in Other Test: {other_test_accuracy:.4f}%')


class TransformerLearning:
    def __init__(self, train_data, test_data, conf):
        """
        Initialize the Transformer learning process.

        Args:
            train_data: Training data
            test_data: Test data
            conf: Configuration parameters
        """
        self.train_data = train_data
        self.test_data = test_data
        self.conf = conf

        self.model = TransformerModel(**self.conf['model_param']).to(self.conf['device'])
        self.loss_fn = nn.CrossEntropyLoss().to(self.conf['device'])
        self.optimizer = optim.SGD(self.model.parameters(), **self.conf['optimizer'])

    def run(self):
        """
        Run the training and testing process.
        """
        loss_train_hist = []
        loss_valid_hist = []
        acc_train_hist = []
        acc_valid_hist = []
        best_acc_valid = 0

        for epoch in range(self.conf['epoch']):
            loss_train, acc_train = self.__train(epoch)
            loss_valid, acc_valid = self.__test()
            loss_train_hist.append(loss_train)
            loss_valid_hist.append(loss_valid)
            acc_train_hist.append(acc_train)
            acc_valid_hist.append(acc_valid)

            if acc_valid > best_acc_valid:
                torch.save(self.model, f'model.pt')
                best_acc_valid = acc_valid
                # self.beast_model = self.model
                print('model update')



            # if acc_valid > 0.4 and self.conf['optimizer']['lr'] >  0.05:
            #     self.conf['optimizer']['lr'] =  0.05
            # if acc_valid > 0.44 and self.conf['optimizer']['lr'] >  0.001:
            #     self.conf['optimizer']['lr'] =  0.001
            # if acc_valid > 0.49 and self.conf['optimizer']['lr'] >  0.0005:
            #     self.conf['optimizer']['lr'] =  0.0005
            # if acc_valid > 0.51 and self.conf['optimizer']['lr'] >  0.0001:
            #     self.conf['optimizer']['lr'] =  0.0001
            # if acc_valid > 0.54 and self.conf['optimizer']['lr'] >  0.00005:
            #     self.conf['optimizer']['lr'] =  0.00005
            # if acc_valid > 0.57 and self.conf['optimizer']['lr'] >  0.000001:
            #     self.conf['optimizer']['lr'] =  0.000001

            logging.info(f'Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}, lr= {self.conf["optimizer"]["lr"]}')

        return loss_train_hist, loss_valid_hist, acc_train_hist, acc_valid_hist

    def __train(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            loss_train: Training loss
            acc_train: Training accuracy
        """
        self.model.train()
        loss_train = AverageMeter()
        acc_train = Accuracy(**self.conf['accuracy'])
        with tqdm(self.train_data, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                if epoch is not None:
                    tepoch.set_description(f"Epoch {epoch}")
                inputs = inputs.to(self.conf['device'])
                targets = targets.to(self.conf['device'])
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_train.update(loss.item())
                acc_train(outputs, targets.int())
                tepoch.set_postfix(loss=loss_train.avg, accuracy=100. * acc_train.compute().item())
        return loss_train.avg, acc_train.compute().item()

    def __test(self):
        """
        Test the model and calculate loss and accuracy.

        Returns:
            loss_valid: Validation loss
            acc_valid: Validation accuracy
        """
        self.model.eval()
        with torch.no_grad():
            loss_valid = AverageMeter()
            acc_valid = Accuracy(**self.conf['accuracy']).to(self.conf['device'])
            for i, (inputs, targets) in enumerate(self.test_data):
                inputs = inputs.to(self.conf['device'])
                targets = targets.to(self.conf['device'])
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_valid.update(loss.item())
                acc_valid(outputs, targets.int())
        return loss_valid.avg, acc_valid.compute().item()



    def validation(self):
        pass


    def check_forward(self,):
        x_batch, y_batch = next(iter(self.train_data))
        outputs = self.model(x_batch.to(self.conf['device']))
        loss = self.loss_fn(outputs, y_batch.to(self.conf['device']))
        print(loss)

    def check_backward(self, epochs):
        self.model.train()
        loss_train = AverageMeter()
        acc_train = Accuracy(**self.conf['accuracy'])
        for epoch in range(epochs):
            with tqdm(self.test_data, unit="batch") as tepoch:
                for inputs, targets in tepoch:
                    if epoch is not None:
                        tepoch.set_description(f"Epoch {epoch}")
                    inputs = inputs.to(self.conf['device'])
                    targets = targets.to(self.conf['device'])
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    loss_train.update(loss.item())
                    acc_train(outputs, targets.int())
                    tepoch.set_postfix(loss=loss_train.avg, accuracy=100.*acc_train.compute().item())

    def plt_loss(self,epoch, loss_train, loss_valid):
        plt.plot(range(epoch), loss_train, 'r-', label='Train')
        plt.plot(range(epoch), loss_valid, 'b-', label='Validation')

        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plt_acc(self,epoch, acc_train, acc_valid):
        plt.plot(range(epoch), acc_train, 'r-', label='Train')
        plt.plot(range(epoch), acc_valid, 'b-', label='Validation')

        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    def __update_learning_rate(self, loss_train_hist, loss_valid_hist, acc_train_hist, acc_valid_hist):
        current_trend = (acc_train_hist[-1] - acc_train_hist[-2]) * 100 / acc_train_hist[-1]
        mean_5past_acc = mean(acc_train_hist[-3:])
        current_to_5past = (acc_train_hist[-1] - mean_5past_acc) * 100 / acc_train_hist[-1]

        if current_trend < -1.5 or current_to_5past < 1:
            self.conf['optimizer']['lr'] = self.conf['optimizer']['lr'] * 0.1
            print('lr: ',self.conf['optimizer']['lr'])

    def select_beast_lr(self, lrs, epochs):
        for lr in lrs:
            self.conf['optimizer']['lr'] = lr
            print(f'LR={lr}')

            lr_model = TransformerModel(**self.conf['model_param']).to(self.conf['device'])
            lr_model.train()
            lr_loss_fn = nn.CrossEntropyLoss().to(self.conf['device'])
            lr_optimizer = optim.SGD(lr_model.parameters(), **self.conf['optimizer'])

            loss_train = AverageMeter()
            acc_train = Accuracy(**self.conf['accuracy'])

            for epoch in range(epochs):
                with tqdm(self.test_data, unit="batch") as tepoch:
                    for inputs, targets in tepoch:
                        if epoch is not None:
                            tepoch.set_description(f"Epoch {epoch}")
                        inputs = inputs.to(self.conf['device'])
                        targets = targets.to(self.conf['device'])
                        outputs = lr_model(inputs)
                        loss = lr_loss_fn(outputs, targets)
                        loss.backward()
                        nn.utils.clip_grad_norm_(lr_model.parameters(), 0.5)
                        lr_optimizer.step()
                        lr_optimizer.zero_grad()
                        loss_train.update(loss.item())
                        acc_train(outputs, targets.int())
                        tepoch.set_postfix(loss=loss_train.avg, accuracy=100.*acc_train.compute().item())
