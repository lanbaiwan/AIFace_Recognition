import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
import torchvision
import csv 
import pandas as pd 
import time


start_time = time.time()  # 记录开始时间


SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def inference(model, img):
    with torch.no_grad():
        y_pred = model(img).sigmoid().flatten().squeeze().cpu().numpy()
    return y_pred

def perpred(path):
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--image' , type=str, default='./test_images/real.png')
    # parser.add_argument('--arch', type=str, default='res50')
    # parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    # opt = parser.parse_args()

    model = get_model('CLIP:ViT-L/14')
    state_dict = torch.load('pretrained_weights/fc_weights.pth', map_location='cpu')
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    stat_from = "imagenet" if 'CLIP:ViT-L/14'.lower().startswith("imagenet") else "clip"

    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    ])

    img_tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).cuda()
    y_pred = inference(model, img_tensor)

    print ("Prediction: ", y_pred)
    if y_pred<0.5:
        return 0
    return 1

def get_all_file_data(directory):
    file_names_without_extension = []  # 初始化一个空列表来存储文件路径
    file_names_with_pred_value = []  # 初始化一个空列表来存储预测数值
    file_data = []  # 初始化返回列表
    i=0
    for root, dirs, files in os.walk(directory):
        for file in files:
            i=i+1
            file_names_with_pred_value.append(perpred(os.path.join(root, file)))  # 获取预测值
            file_name_without_extension = os.path.splitext(file)[0]
            file_names_without_extension.append(file_name_without_extension)  # 获取文件名称
            if(i==10):
                i=0
                proccess_time = time.time()  # 记录时间
                print(f"程序运行时间：{proccess_time - start_time} 秒")

    file_data.append(file_names_without_extension)
    file_data.append(file_names_with_pred_value)# 整合两个列表
    return file_data

def write_to_csv(file_list, output_file_path):

    file_names, numbers = file_list

    # 确保两个子列表长度相同
    if len(file_names) != len(numbers):
        raise ValueError("文件名列表和数字列表的长度必须相同。")

    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 写入数据行
        for file_name, number in zip(file_names, numbers):
            writer.writerow([file_name, number])

if __name__ == '__main__':
    
    irectory_path = '/home/data/szk/face/face'
    all_file_data = get_all_file_data(irectory_path)
    write_to_csv(all_file_data, '/home/data/szk/face/cla_pre.csv')
    end_time = time.time()  # 记录结束时间
    print(f"程序运行时间：{end_time - start_time} 秒")



    
