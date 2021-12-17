import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RockImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length
    
    
transforms_train = transforms.ToTensor()
transforms_val = transforms.ToTensor()
transforms_test = transforms.ToTensor()

param_epoch = 10
param_batch = 64
param_learning_rate = 0.00001

train_data_set = RockImageDataset(data_path="./train", transforms=transforms_train)
val_data_set = RockImageDataset(data_path="./val", transforms=transforms_val)
test_data_set = RockImageDataset(data_path="./test", transforms=transforms_test)

train_loader = DataLoader(train_data_set, batch_size=param_batch, shuffle=True)
val_loader = DataLoader(val_data_set, batch_size=param_batch, shuffle=False)
test_loader = DataLoader(test_data_set, batch_size=param_batch, shuffle=False)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()
    
    
num_classes = train_data_set.num_classes

custom_model = models.resnet152(pretrained=True)
custom_model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
custom_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=param_learning_rate)

train_losses = []
valid_losses = []
train_acc = []
valid_acc = []

since = time.time()
best_model_wts = copy.deepcopy(custom_model.state_dict())
best_acc = 0.0


for epoch in range(1, param_epoch + 1):
    
    train_loss = 0.0
    valid_loss = 0.0
    train_corrects = 0
    valid_corrects = 0
    
    # train
    custom_model.train()
    for batch, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)
        

        optimizer.zero_grad()
        
        output = custom_model(images)
        
        _, preds = torch.max(output, 1)
        
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        train_corrects += torch.sum(preds == labels.data)
        
        del images
        del labels
        del loss
        
    
    # validate
    with torch.no_grad():
        
        custom_model.eval()
    
        for batch, item in enumerate(val_loader):
        
            images = item['image'].to(device)
            labels = item['label'].to(device)
        
            output = custom_model(images)
        
            _, preds = torch.max(output, 1)
        
            loss = criterion(output, labels)
        
            valid_loss += loss.item() * images.size(0)
            valid_corrects += torch.sum(preds == labels.data)
        
            del images
            del labels
            del loss
        
    # calculate loss & accuracy
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(val_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    train_corrects = train_corrects.double()/len(train_loader.sampler)
    valid_corrects = valid_corrects.double()/len(val_loader.sampler)
    train_corrects = train_corrects.item()
    valid_corrects = valid_corrects.item()
    
    
    train_acc.append(train_corrects)
    valid_acc.append(valid_corrects)
    
    if valid_corrects > best_acc:
        best_acc = valid_corrects
#         best_model_wts = copy.deepcopy(custom_model.state_dict())
        
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    print('Epoch: {} \tTraining Acc: {:.6f} \tValidation Acc: {:.6f}'.format(
        epoch, train_corrects, valid_corrects))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:6f}'.format(best_acc))

# custom_model.load_state_dict(best_model_wts)

plt.figure(figsize=(11, 3.5))
plt.subplot(121)
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)

plt.subplot(122)
plt.plot(train_acc, label='Training accuracy')
plt.plot(valid_acc, label='Validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(frameon=False)

plt.show()


test_loss     = 0.0
class_correct = [0]*6
class_total   = [0]*6

custom_model.eval()

conf_matrix = np.zeros((6,6))

for item in test_loader:
    images = item['image'].to(device)
    labels = item['label'].to(device)
    outputs = custom_model(images)
    
    loss = criterion(outputs, labels)
    
    test_loss += loss.item()*images.size(0)
    
    _, pred = torch.max(outputs, 1)
    
    correct_tensor = pred.eq(labels.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())
    
    
    for i in range(labels.size(0)):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        
        conf_matrix[label][pred.data[i]] += 1
        
        
conf_matrix_sum = conf_matrix.sum(axis=1)
conf_matrix_prob = np.zeros((6,6))

for i in range(6):
    conf_matrix_prob[i] = conf_matrix[i]/conf_matrix_sum[i]
    
x_axis_labels = os.listdir('./test')
y_axis_labels = os.listdir('./test')

plt.subplots(figsize=(10,8))
ax = sns.heatmap(conf_matrix_prob, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, fmt='.2f',annot_kws={"size": 12}, 
                vmax=1, cmap='Blues', linewidth=1, square=True)

ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(labels=ax.get_yticklabels(), va='center', rotation=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()
