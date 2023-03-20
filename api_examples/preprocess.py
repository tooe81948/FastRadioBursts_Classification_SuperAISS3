import numpy as np
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import (Compose, 
                                    Resize, 
                                    ToTensor, 
                                    Normalize, 
                                    ToPILImage,
                                    Lambda)

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std),
])

def preprocess_image(sub_signal):
    sub_signal = Image.fromarray(sub_signal, "L").convert("RGB")    
    return transform(sub_signal).unsqueeze(0)


def preprocess_signal(sub_signal):
    sub_signal = sub_signal - np.median(sub_signal, axis=0, keepdims=True)
    sub_signal = sub_signal / np.std(sub_signal, axis=0, keepdims=True)
    # sub_signal = (sub_signal - np.min(sub_signal)) / (np.max(sub_signal) - np.min(sub_signal))
    return sub_signal

def preprocess_data(npy):
    return preprocess_image(preprocess_signal(npy))
    


# if __name__ == "__main__":
#     from models import TimmLightningModel
#     st = time.time()
#     # model
#     model_path = "/home/superai052/super_workspace/weights/coatnet_2_rw_224_e30.pth"
#     model_name = "coatnet_2_rw_224"
#     num_classes = 3
#     model_Timm = TimmLightningModel()
#     threshold = 0.5
#     src = '/lustrefs/disk/project/lt900038-ai23tn/frb_data/train/B0531+21_58713_43190_reduced_fc_0032130.npy'
#     npy = np.load(src)
#     data = preprocess_data(npy)
#     prob = model_Timm.predict(data)
#     prob = (prob > threshold).int()
#     prob = "".join(prob[0].numpy().astype(str))
#     print(prob) 
#     print(time.time() - st)