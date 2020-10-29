import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class CNNCell(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNCell, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              kernel_size=3,
                              out_channels=output_channels)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU()

    def forward(self, batch_data):
        output = self.conv(batch_data)
        output = self.bn(output)
        output = self.relu(output)

        return output


class Dataset:
    """
    build a map-style dataset
    """
    def __init__(self,data,targets,transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if self.transform == None:
            return self.data[idx],self.targets[idx]
        else:
            return self.transform(self.data[idx]),self.targets[idx]

class TestDataset:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]


class CNNNetwork(nn.Module):
    def __init__(self, learning_rate, batch_size, n_classes, epochs):
        super(CNNNetwork, self).__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.loss_history = []
        self.accuracy_history = []

        self.cell1 = CNNCell(input_channels=1, output_channels=32)
        self.cell2 = CNNCell(input_channels=32, output_channels=32)
        self.cell3 = CNNCell(input_channels=32, output_channels=32)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.cell4 = CNNCell(input_channels=32, output_channels=64)
        self.cell5 = CNNCell(input_channels=64, output_channels=64)
        self.cell6 = CNNCell(input_channels=64, output_channels=64)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.network = nn.Sequential(self.cell1, self.cell2, self.cell3, self.max_pool1,
                                     self.cell4, self.cell5, self.cell6, self.max_pool2)

        self.network_output_dims = self.calculate_input_dims()
        self.fc = nn.Linear(in_features=self.network_output_dims, out_features=n_classes)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)
        self.get_data()

    def forward(self, batch_data):
        batch_data = batch_data.clone().detach().requires_grad_(True).to(self.device)
        output = self.network(batch_data)
        output = output.view(-1, self.network_output_dims)
        output = self.fc(output)

        return output

    def calculate_input_dims(self):
        batch_data = T.zeros((1, 1, 28, 28))
        batch_data = self.cell1.forward(batch_data)
        batch_data = self.cell2.forward(batch_data)
        batch_data = self.cell3.forward(batch_data)

        batch_data = self.max_pool1(batch_data)

        batch_data = self.cell4.forward(batch_data)
        batch_data = self.cell5.forward(batch_data)
        batch_data = self.cell6.forward(batch_data)

        batch_data = self.max_pool2(batch_data)

        return int(np.prod(batch_data.size()))

    def get_data(self):
        root = "/home/ss12852"
        # 42000 train 784 pixels + 1 label (take a while to load)
        train_data = np.loadtxt(os.path.join(root,"train.csv"),delimiter=",",skiprows=1)
        # 28000 test 784 pixels
        test_data = np.loadtxt(os.path.join(root,"test.csv"),delimiter=",",skiprows=1)

        transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomAffine(degrees=15,translate=(1/7,1/7),shear=15),
                                transforms.RandomRotation(degrees=15),
                                transforms.ToTensor()])

        x_train = train_data[:,1:].reshape(-1,28,28).astype(np.uint8)
        y_train = T.LongTensor(train_data[:,0])
        train_dataset = Dataset(x_train,y_train,transform)

        x_test = test_data.reshape(-1,1,28,28) # pytorch channel first
        x_test = T.Tensor(x_test)/255.
        test_dataset = TestDataset(x_test, transform)

        mnist_train_data = MNIST('mnist/', train=True, download=True, transform=ToTensor())

        self.train_data_loader = T.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.mnist_train_data_loader = T.utils.data.DataLoader(mnist_train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        #mnist_test_data = MNIST('mnist/', train=False, download=True, transform=ToTensor())

        self.test_data_loader = T.utils.data.DataLoader(test_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True)

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                loss = self.loss(prediction, label)
                self.accuracy_history.append(acc)
                ep_loss += loss.item()
                ep_acc.append(acc.item())
                loss.backward()
                self.optimizer.step()
            for j, (input, label) in enumerate(self.mnist_train_data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                loss = self.loss(prediction, label)
                self.accuracy_history.append(acc)
                ep_loss += loss.item()
                ep_acc.append(acc.item())
                loss.backward()
                self.optimizer.step()
            print('Finished Epoch ', i, 'Total Loss %.3f Training Accuracy %.3f' % (ep_loss, np.mean(ep_acc)))
            self.loss_history.append(ep_loss)

    def _test(self):
        self.eval()
        test_pred = []
        for j, input in enumerate(self.test_data_loader):
            prediction = self.forward(input)
            classes = T.argmax(prediction, dim=1)
            test_pred.append(classes.detach().cpu().numpy())
        test_pred = np.concatenate(test_pred)

        imageid = pd.Series(np.arange(len(test_pred)))+1
        df = pd.DataFrame({"ImageId":imageid,"Label":test_pred})
        df.set_index("ImageId")
        df.to_csv("/home/ss12852/test_pred.csv",index=False)


if __name__ == '__main__':
    network = CNNNetwork(learning_rate=0.001, batch_size=32, epochs=30, n_classes=10)
    network._train()
    plt.plot(network.loss_history)
    plt.show()
    plt.plot(network.accuracy_history)
    plt.show()
    network._test()
