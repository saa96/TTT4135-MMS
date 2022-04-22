#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {} if device=='cpu' else {'num_workers': 1, 'pin_memory': True}
batch_size=4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# Splitting the training set into train and validation sets.
trainset, valset = train_test_split(trainset, test_size=0.15, random_state=1)

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

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Network structuring inspired by assignments in TDT4265,
        # I feel that this structuring is clearer

        # Feature extraction layers
        self.num_classes = len(classes)
        self.feat_extractor = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(128)
        )
        self.num_output_features = 4*4*128
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 120),
            nn.Linear(120,84),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.feat_extractor(x)
        x = x.view(batch_size,-1)
        x = self.classifier(x)
        return x

def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    accuracy = 0
    loss = 0
    num_batches = 0
    with torch.no_grad():
        for (X_batch, Y_batch) in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # Forward pass the images through our model
            output_probs = model(X_batch)
            # Compute Loss and Accuracy
            pred_val = torch.argmax(output_probs, dim = 1)
            hit = Y_batch.eq(pred_val).float()
            loss += loss_criterion(output_probs, Y_batch).item()
            accuracy += torch.mean(hit).item()
            num_batches += 1
    
    average_loss = loss/num_batches
    accuracy /= num_batches

    return average_loss, accuracy


def do_early_stop(val_loss_hist, early_stop_count):
    if len(val_loss_hist) < early_stop_count: 
        return False
    
    # We only care about the last [early_stop_count] losses.
    relevant_loss = val_loss_hist[-early_stop_count:]
    first_loss = relevant_loss[0]
    if first_loss == min(relevant_loss):
        print("Early stop criteria met")
        return True
    return False

val_hist_loss = []
val_hist_acc = []
train_hist_loss = []
train_hist_acc = []
early_stop_count = 3

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
number_of_epochs = 10
for epoch in range(number_of_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    hit = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # To show image
        show_img = False
        if show_img:
            img = inputs.cpu()
            img = img[0][0] / 2 + 0.5     # unnormalize
            plt.imshow(img)
            plt.show()
            if i > 10:
                break

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    train_loss = running_loss / i
    print('[%d] training loss: %.3f' % (epoch + 1, train_loss))

    # validation step
    #val_acc = 0
    #val_loss = 0
    #val_tot = 0
    net.eval()
    val_loss, val_acc = compute_loss_and_accuracy(valloader, net, criterion)
    val_hist_loss.append(val_loss)
    val_hist_acc.append(val_acc)
    train_hist_loss.append(train_loss)

    print('[{}] validation loss: {}, validation accuracy: {}'.format(epoch + 1, val_loss, val_acc))


    if do_early_stop(val_hist_loss, early_stop_count):
        print("Early stopp")
        break

# Printing test loss and accuracy
test_loss, test_acc = compute_loss_and_accuracy(testloader, net, criterion)

print("Test loss: {}, test accuracy: {}".format(test_loss,test_acc))

x = np.linspace(0,len(val_hist_loss),len(val_hist_loss))
plt.figure()
plt.plot(x,val_hist_loss,x,train_hist_loss)
plt.title("Cross entropy loss")
plt.legend(['Validation loss', 'Training Loss'])
plt.xlabel('Epochs'), plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(x,val_hist_acc)
plt.title("Accuracy")
plt.legend(['Validation accuracy'])
plt.xlabel('Epochs'), plt.ylabel("Accuracy")
plt.show()


# Plotting of PR-curve

# Copied implementation for finding PR curve from Sebastian Skogen Raa and his group
# As I was struggeling with getting it to work
from sklearn.metrics import precision_recall_curve, auc

y_pred = []
y_test = []
accuracy = 0

with torch.no_grad():
        for (X_batch, Y_batch) in trainloader:
            
            # Transfer images/labels to GPU VRAM, if possible
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # Forward pass the images through our model
            outputs =net(X_batch)
           
           # Add predicted values to list
            for pred in outputs.cpu().detach().numpy():
                y_pred.append(pred.tolist())
        
            # Add labels to list
            for label in Y_batch.cpu().detach().numpy():
              test = [0]*10
              test[label]=1
              
              y_test.append(test)

y_pred = np.array(y_pred)
y_test = np.array(y_test)

precision = dict()
recall = dict()
for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:,i], y_pred[:, i])

fig, ax = plt.subplots()
for i in range(10):
  ax.plot(recall[i], precision[i],label = classes[i])
  print(f'Accuracy for class: {classes[i]} is', auc(recall[i], precision[i]))

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.legend()
plt.show()

'''
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


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
'''


