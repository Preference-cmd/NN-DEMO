import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class MnistModel(torch.nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        y_pred = self.activate(self.linear1(x))
        y_pred = self.activate(self.linear2(y_pred))
        y_pred = self.activate(self.linear3(y_pred))
        y_pred = self.activate(self.linear4(y_pred))
        y_pred = self.linear5(y_pred)

        return y_pred


class Script:
    def __init__(self, filepath):  # initialize the components
        self.batch_size = 128
        self.transform = transforms.Compose([
            # the original images should be transformed to torch.tensor
            transforms.ToTensor(),
            # maps the original distribution to the standard normal distribution
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_loader, self.test_loader = self.get_data(filepath)
        self.model = MnistModel().cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        self.accuracy = []

    def get_data(self, filepath):
        train_dataset = datasets.MNIST(root=filepath, train=True, transform=self.transform, download=True)
        test_set = datasets.MNIST(root=filepath, train=False, transform=self.transform, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self, epoch):
        running_loss = 0.0
        for batch_index, data in enumerate(self.train_loader, 0):
            inputs, target = data
            inputs = inputs.cuda()
            target = target.cuda()
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if batch_index % 300 == 299:
                print(f'{epoch+1} {batch_index} loss: {running_loss / 300}')
                running_loss = 0.0

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100*correct/total
            print(f'Accuracy on test set: {accuracy}%')
            self.accuracy.append(accuracy)


def main():  # training loops
    scripts = Script(r'.\dataset')
    for epoch in range(15):
        scripts.train(epoch)
        scripts.test()
    plt.title('Accuracy on test set')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(range(1, 16), scripts.accuracy)
    plt.show()


if __name__ == '__main__':
    main()
