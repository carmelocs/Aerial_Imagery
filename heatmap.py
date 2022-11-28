import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt 
from torchvision.transforms.functional import crop

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", help="location of image")
# parser.add_argument("--outdir", help="CAM images are stored in this folder")
parser.add_argument("--crop_size", type=int, help="size to crop in original image")
args = parser.parse_args()

image_path = args.image_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE}...")

CROP_SIZE = args.crop_size

image_pil = Image.open(image_path)
image_pil.save('test_heatmap.jpg')
# print(image_pil.format, image_pil.size, image_pil.mode)
width, height = image_pil.size
print(width, height)

net = models.resnet18()
net.fc = nn.Linear(net.fc.in_features, 2)

net.load_state_dict(torch.load("./best_model_resnet18.pt", map_location=DEVICE))
net.eval()

img = np.zeros((width, height), dtype=np.uint8)

preprocess = transforms.ToTensor()

for cx in range(0, int(width - CROP_SIZE/2), 1):
    for cy in range(0, int(height - CROP_SIZE/2), 1):
        image_crop = crop(image_pil, cx, cy, CROP_SIZE, CROP_SIZE)
        logit = net(preprocess(image_crop).unsqueeze(0))
        h_x = F.softmax(logit, dim=1).data.squeeze()
        prob = h_x[1].item()
        pixel = int(prob * 255)
        img[int(cx + CROP_SIZE/2)][int(cy + CROP_SIZE/2)] = pixel
        
print("output heatmap.jpg")

img_crop = img[int(CROP_SIZE/2):, int(CROP_SIZE/2):]

cam = cv2.resize(img_crop, (width, height))
heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)

image_cv = cv2.imread(image_path)
result = cv2.addWeighted(image_cv, 0.3, heatmap, 0.5, 0)
cv2.imwrite('heatmap.jpg', heatmap)
cv2.imwrite('result.jpg', result)