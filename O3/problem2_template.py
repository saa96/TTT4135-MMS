#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sklearn



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {} if device=='cpu' else {'num_workers': 1, 'pin_memory': True}
batch_size=4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# Splitting the training set into train and validation sets.
trainset, valset = train_test_split(trainset, test_size=0.2, random_state=1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

#print(trainset)
print("Len of train={} and len of val={}".format(len(trainset),len(valset)))

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Network structuring inspired by assignments in TDT4265,
        # I feel that this structuring is clearer
        # Conv layers
        self.num_classes = len(classes)
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.num_output_features = 4*4*16
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 120),
            nn.Linear(120,84),
            nn.Linear(84, self.num_classes)
        )

        """ 
        self.conv1 = nn.Conv2d(
                    in_channels=3, 
                    out_channels=6, 
                    kernel_size=5)
        self.pool  = nn.MaxPool2d(
                    kernel_size=2, 
                    stride=2)
        self.conv2 = nn.Conv2d(
                    in_channels=6, 
                    out_channels=16, 
                    kernel_size=5
        )
        self.conv3 = nn.Conv2d(
                    in_channels=16, 
                    out_channels=16, 
                    kernel_size=5,
                    padding=2
        )
        # Size of last conv layer
        self.conv_layer_size = 16 * 5 * 5
        # Classification layer
        self.fc1 =   nn.Linear(
                    in_features = 64,
                    out_features = 120)
        self.fc2 =   nn.Linear(
                    in_features = 120, 
                    out_features = 84)
        self.fc3 =   nn.Linear(
                    in_features = 84, 
                    out_features = 10) """

    def forward(self, x):
        batch_size = x.shape[0]
        #x = self.pool(nn.ReLU(self.conv1(x)))
        #x = self.pool(nn.ReLU(self.conv2(x)))
        #x = self.pool(nn.ReLU(self.conv3(x)))
        #x = self.pool(nn.ReLU(self.conv3(x)))
        x = self.feat_extractor(x)
        x = x.view(batch_size,-1)
        x = self.classifier(x)
        #x = nn.ReLU(self.fc1(x))
        #x = nn.ReLU(self.fc2(x))
        #x = self.fc3(x)
        return x


net = Net()
net.to(device)

train_hist = dict(loss=float, accuracy=float)
val_hist   = dict(loss=float, accuracy=float)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(model:        nn.Module, 
          train_loader: torch.utils.data.DataLoader,
          optimizer: optim.SGD):

          return 0

def do_early_stop(val_loss, val_loss_hist):
    if val_loss >= val_loss_hist[-5]: 
        print("Early stopp kicked in.")
        return True
    else: return False

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        #print("i: {}, data: {}".format(i,data))
        inputs, labels = data[0].to(device), data[1].to(device)
        print(": {}, data: {}".format(inputs,labels))
        break

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # validation step
        #net.eval()

        #if do_early_stop(val_loss, val_loss_hist):
        #    break

        # print statistics
        running_loss += loss.item()
    #print('[%d] loss: %.3f' %
    #        (epoch + 1, running_loss / i))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%'
      % (100 * correct / total))
