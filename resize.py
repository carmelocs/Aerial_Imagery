import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision.transforms.functional import crop
from tqdm import tqdm
import os 

file_dir = os.getcwd() + '/data/'

parser = argparse.ArgumentParser()
parser.add_argument("--resize", type=int, help="size to compress original image", default=1)
parser.add_argument("--file_name", help="file name of image")

args = parser.parse_args()
print(args)

# image_path = args.image_path
file_name = args.file_name
image_path = file_dir + file_name

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE}...")

RESIZE = args.resize

image_pil = Image.open(image_path)
width, height = image_pil.size
print(f"Image width: {width} \nImage height: {height}")

image_cv = cv2.imread(image_path)

if (RESIZE == 1):
    cv2.imwrite(file_dir + file_name[:-4] + '_result.jpg', image_cv)
else:
    width_resize, height_resize = int(width/RESIZE), int(height/RESIZE)
    image_resize = cv2.resize(image_cv, (width_resize, height_resize))
    cv2.imwrite(file_dir + file_name[:-4] + "_resize.jpg", image_resize)
    print(f"Resized image width: {width_resize}, height: {height_resize}")