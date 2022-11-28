import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", help="location of image")
# parser.add_argument("--outdir", help="CAM images are stored in this folder")
parser.add_argument()
args = parser.parse_args()

image_path = args.image_path
# output_dir = args.outdir

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE}...")

IMAGE_SIZE = 224
UPSAMPLE_SIZE = 256

net = models.resnet18()
net.fc = nn.Linear(net.fc.in_features, 2)

net.load_state_dict(torch.load("./best_model_resnet18.pt", map_location=DEVICE))
finalconv_name = 'layer4'
net.eval()

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters()) # 将参数变换为列表
weight_softmax = np.squeeze(params[-2].data.numpy()) # 提取softmax层的参数

def returnCAM(feature_conv, weight_softmax, class_idx, positive_prob):
    # generate the class activation maps upsample to 256x256
    size_upsample = (UPSAMPLE_SIZE, UPSAMPLE_SIZE)
    bc, nc, h, w = feature_conv.shape
    output_cam = []
    # class_idx为预测分数较大的类别数字表示的数组，一张图片中有N个类物体，则数组中N个元素
    for idx in class_idx:
        # 回到GAP的值
        # weight_softmax中预测为第idx类的参数w乘以feature_map,为了相乘reshape map的形状
        cam = weight_softmax[idx].dot(feature_conv.reshape(nc, h * w))
        #将feature_map的形状reshape回去
        cam = cam.reshape(h, w)
				
		# 归一化操作（最小值为0，最大值为1）
        # np.min 返回数组的最小值或沿轴的最小值
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
			
		# 转换为图片255的数据
        # np.uint8() Create a data type object.
        cam_img = np.uint8(255 * cam_img * positive_prob)

		# resize 图片尺寸与输入图片一致
        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam

# 数据处理，先缩放尺寸到(224,224)，再变换数据类型为tensor,最后normalize
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])

img_pil = Image.open(image_path)
img_pil.save('test_cam.jpg')
# 将图片数据处理成所需要的可用数据 tensor
img_tensor = preprocess(img_pil)
# 处理图片为Variable数据
img_variable = Variable(img_tensor.unsqueeze(0))
#将图片输入网络得到预测类别分数
logit = net(img_variable)

h_x = F.softmax(logit, dim=1).data.squeeze()
# 对分类的预测类别分数排序，输出预测值和在列表中的位置
idx = 1
prob = h_x[1]

# 转换数据类型
prob = prob.numpy()
idx = np.array(idx)

# 输出与图片尺寸一致的CAM图片
# generate class activation mapping for the positive prediction, i.e., idx equals 1
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx], prob)

# render the CAM and output
print("output CAM.jpg")

img = cv2.imread(image_path)
height, width, _ = img.shape
# 生成热力图(single image)
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
# # 生成热力图(for different images)
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0] * prob, (width, height)), cv2.COLORMAP_JET)
result = cv2.addWeighted(img, 0.3, heatmap, 0.5, 0)
cv2.imwrite('heatmap_cam.jpg', heatmap)
cv2.imwrite('CAM.jpg', result)