# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
import os
import cv2
from PIL import Image

import utils

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

model_dir = "./output/resnet_finetune_2019-07-02-21-12-03/best_val_model.pth"

# img_dir = "./data/excavator_cls/val/excavator/"
img_root = "./data/excavator_cls/val/excavator/"


num_classes = 2

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#################################
# Prepare model
#################################


# Initialize the model for this run
model_ft, input_size = utils.initialize_model(model_name, num_classes, False, use_pretrained=False)

model_ft.load_state_dict(torch.load(model_dir))

model_ft.eval()

# Send the model to GPU
model_ft = model_ft.to(device)

input_size = 224

#################################
# Prepare data
#################################

# Just normalization for validation
trans = transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])




#################################
# Inference
#################################

# img_names = os.listdir(img_root)

# for img_name in img_names:
#     img_dir = os.path.join(img_root, img_name)

#     img = Image.open(img_dir)

#     img_input = trans(img)
#     img_input = torch.unsqueeze(img_input, dim=0)
#     img_input = img_input.to(device)

#     output = model_ft(img_input)

#     output = nn.functional.softmax(output, dim=1)

#     print(img_name, output)




img_names = os.listdir(img_root)

for img_name in img_names:
    img_dir = os.path.join(img_root, img_name)

    img = cv2.imread(img_dir)
    img = img[:,:,::-1]
    img = cv2.resize(img, (input_size, input_size))

    img_input = trans(img)
    img_input = torch.unsqueeze(img_input, dim=0)
    img_input = img_input.to(device)

    output = model_ft(img_input)

    output = nn.functional.softmax(output, dim=1)

    print(img_name, output)




