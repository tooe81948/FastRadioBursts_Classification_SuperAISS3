import cv2
import torch
import time
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


# if __name__ == "__main__":
#     st = time.time()
#     sub_signal = np.zeros([2560, 2560], dtype=np.uint8)
#     sub_signal = preprocess_image(sub_signal)
#     print(type(sub_signal), sub_signal.size())
#     print(time.time() - st)










# from models import TimmLightningModel
# import numpy as np
# import time

# def preprocess_signal(sub_signal):
#     sub_signal = sub_signal - np.median(sub_signal, axis=0, keepdims=True)
#     sub_signal = sub_signal / np.std(sub_signal, axis=0, keepdims=True)
#     # sub_signal = (sub_signal - np.min(sub_signal)) / (np.max(sub_signal) - np.min(sub_signal))
#     return sub_signal

# if __name__ == "__main__":
#     st = time.time()
#     src = '/lustrefs/disk/project/lt900038-ai23tn/frb_data/train/B0531+21_58713_43190_reduced_fc_0001023.npy'
#     npy = np.load(src)
#     sub_signal = preprocess_image(preprocess_signal(npy))
#     print(sub_signal.shape)
#     print(time.time() - st)