import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F
import argparse
# import matplotlib.pyplot as plt
from torchvision.transforms.functional import crop
from tqdm import tqdm
import os 

file_dir = os.getcwd() + '/data/'

parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", help="location of image")
# parser.add_argument("--outdir", help="CAM images are stored in this folder")
parser.add_argument("--crop_size", type=int, help="size to crop in original image", default=224)
parser.add_argument("--step", type=int, help="size to crop in original image", default=60)
parser.add_argument("--file_name", help="file name of image")

args = parser.parse_args()
print(args)

# image_path = args.image_path
file_name = args.file_name
image_path = file_dir + file_name

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE}...")

CROP_SIZE = args.crop_size
STEP = args.step

image_pil = Image.open(image_path)
# image_pil.save('test_heatmap.jpg')
# print(image_pil.format, image_pil.size, image_pil.mode)
width, height = image_pil.size
print(f"Image width: {width} \nImage height: {height}")

net = models.resnet18()
net.fc = nn.Linear(net.fc.in_features, 2)

net.load_state_dict(torch.load("./best_model_resnet18.pt", map_location=DEVICE))
net.eval()

net.to(DEVICE)

img = np.zeros((height + CROP_SIZE, width + CROP_SIZE), dtype=np.float32)

preprocess = transforms.ToTensor()

for cx in tqdm(range(0, height, 30)):
    for cy in range(0, width, 30):
        image_crop = crop(image_pil, cx, cy, CROP_SIZE, CROP_SIZE)
        logit = net(preprocess(image_crop).unsqueeze(0).to(DEVICE))
        h_x = F.softmax(logit, dim=1).data.squeeze()
        prob = h_x[1].item()
        img[cx : cx + CROP_SIZE, cy : cy + CROP_SIZE] += prob
        
print("output heatmap.jpg")

img_crop = img[:-CROP_SIZE, :-CROP_SIZE]
img_crop = np.uint8(img_crop/img_crop.max() * 255)

cam = cv2.resize(img_crop, (width, height))
heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)

image_cv = cv2.imread(image_path)
result = cv2.addWeighted(image_cv, 0.5, heatmap, 0.3, 0)
# cv2.imwrite('heatmap.jpg', heatmap)
cv2.imwrite(file_dir + file_name[:-4] + '_result.jpg', result)