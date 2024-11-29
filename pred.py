import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from models import get_model
from PIL import Image
import random
import argparse

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




def init_pred():
    # 手动设置参数
    args = {
        'arch': 'CLIP:ViT-L/14',  # 你的模型架构
        'ckpt': './pretrained_weights/11_11_final_TTO_1.pth'  # 你的预训练权重路径
    }
    # 将字典转换为Namespace对象
    opt = argparse.Namespace(**args)
    # 获取模型
    model = get_model(opt.arch,opt)
    # 加载预训练权重

    checkpoint = torch.load(opt.ckpt, map_location='cpu')
    state_dict = checkpoint['model']

    model.fc.load_state_dict(state_dict)

    model.eval()  # 设置为评估模式

    stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"

    # 定义转换
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
    ])
    return transform,model
def get_pred(image_path,transform,model):
    # 读取图像，应用转换
    img_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

    # 执行推理
    y_pred = inference(model, img_tensor)

    print("Prediction: ", y_pred)
    return y_pred

if __name__=='__main__':
    transform,model=init_pred()
    i=0
    while(i<10):
        i+=1
        get_pred(r'C:\Users\79173\Desktop\zhuomian\lanbaiwan.jpg',transform,model)
