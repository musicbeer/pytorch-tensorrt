import torch
from torch.autograd import Variable
import tensorrt as trt
from tensorrt.parsers import onnxparser
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import os
import numpy as np
from argparse import ArgumentParser

from resnet import resnet50
import time
import cv2
from torchvision import transforms
from trt_engine import trt_engine
import pdb
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def read_image_chw(img_path, width, height):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (width, height))

    im = transform(img)
    im = im.numpy()
    return im

args = ArgumentParser().parse_args()
args.input_size = 224
args.input_channel = 3
args.fc_num = 1000
args.batch_size = 4
args.trt_model_name = "resnet50_32.trt"
args.testdir='data'
engine = trt_engine('resnet',args.trt_model_name).build_engine()
classes=os.listdir(args.testdir)
total = 0
correct = 0
start=time.time()
for i in range(len(classes)):
    images=os.listdir(os.path.join(args.testdir,classes[i]))
    for image in images:
        gt=int(i)
        img = read_image_chw(os.path.join(args.testdir,classes[i],image),
                args.input_size, args.input_size)
        output = engine.infer(img)
        print(np.squeeze(output[0])[:5])
        conf, pred = torch.Tensor(np.squeeze(output[0])).topk(1, dim=0)
        pred = int(pred.data[0])
        # print('gt:',classes[i],'pred:',classes[pred])
        # if pred == gt:
        #     correct += 1
        # total += 1
end=time.time()
print('trt:',end-start)

# speed test 

model = resnet50(pretrained=True)
model.cuda()
model.eval()
trans=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
classes=os.listdir(args.testdir)
total = 0
correct = 0
start=time.time()
for i in range(len(classes)):
    images=os.listdir(os.path.join(args.testdir,classes[i]))
    for image in images:
        gt=int(i)
        img = cv2.imread(os.path.join(args.testdir,classes[i],image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.input_size, args.input_size))
        img = transform(img)
        img=torch.unsqueeze(img, 0)

        batch_x=Variable(img)
        batch_x=batch_x.cuda()
        out = model(batch_x)
        print(out.data[0][:5])
        _,pred=torch.max(out.data, 1)
        pred = int(pred)
        # print('gt:',classes[i],'pred:',classes[pred])
        # if pred == gt:
        #     correct += 1
        # total += 1
end=time.time()
print('pytorch:',end-start)
