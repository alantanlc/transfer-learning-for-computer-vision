from __future__ import print_function, division

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import copy
from CheXpert import *

plt.ion()   # interactive mode

# Generic function to train model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                inputs = sample_batched['image'].float().to(device)
                labels = sample_batched['pathologies'].float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Compute subset accuracy
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                    y_pred = preds.cpu().numpy()
                    y_true = labels.cpu().numpy()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += np.sum(y_true == y_pred)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / (dataset_sizes[phase] * 14)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Data augmentation and normalization for training
# Just normalization for validation
mean, std = 127.8989, 74.69748171138374
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(365),
        transforms.RandomCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(365),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
}

root_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/'
csv_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/CheXpert-v1.0-small/'

# root_dir = '~/projects/ce7454-grp17/data/'
# csv_dir = '~/projects/ce7454-grp17/data/CheXpert-v1.0-small/'

# csv_dir = './'

image_datasets = {x: CheXpertDataset(csv_file=os.path.join(csv_dir, x + '.csv'), root_dir=root_dir, transform=data_transforms[x]) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'valid']}
print('dataset_sizes:', dataset_sizes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device, '\n')

# ConvNet as a fixed feature extractor
model_conv = models.resnet18(pretrained=True)

# Freeze parameters so that gradients are not computed in backward()
# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have required_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model_conv.fc = nn.Linear(num_ftrs, 14)

optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.001)

# Send model to device
model_conv = model_conv.to(device)

criterion = nn.BCEWithLogitsLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

print('End of program')