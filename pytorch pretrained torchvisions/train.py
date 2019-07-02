# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import time
import os
import logging


import utils

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


#################################
# Hyper-paras
#################################

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data/excavator_cls"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 64

base_lr = 0.001

step_size = 20

gamma = 0.1

cls_weight = [3, 1] ### 3 for excavator and 1 for others, because the img numbers are unbalanced

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

if feature_extract: 
    output_dir = os.path.join("./output/", model_name + time.strftime('_extract_%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
else:
    output_dir = os.path.join("./output/", model_name + time.strftime('_finetune_%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

log_dir = os.path.join(output_dir, "log.log")


#################################
# Initialize logger
#################################

logging.basicConfig(level=logging.INFO,
                    filename=log_dir,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

console = logging.StreamHandler()
logger.addHandler(console)


#################################
# Build models
#################################

# Initialize the model for this run
model_ft, input_size = utils.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)



#################################
# Prepare dataset
#################################

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# print(image_datasets['train'].class_to_idx)
# os._exit()

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#################################
# Create the optimizer
#################################

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=base_lr, momentum=0.9)

scheduler = StepLR(optimizer_ft, step_size=step_size, gamma=gamma)




#################################
# Set up training
#################################

# Setup the loss fxn
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(cls_weight).to(device))

# Train and evaluate
model_ft, hist = utils.train_model(model_ft, \
    dataloaders_dict, criterion, optimizer_ft, device, scheduler, logger, output_dir, \
    num_epochs=num_epochs, is_inception=(model_name=="inception"))










