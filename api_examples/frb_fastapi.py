
import numpy as np
import json
import uvicorn
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from models import TimmLightningModel
from preprocess import preprocess_image
from fastapi.responses import PlainTextResponse

# model
model_path = "/home/superai052/super_workspace/model_suea/weights/gluon_resnet18_v1b/gluon_resnet18_v1b_e10.pth"
model_name = "gluon_resnet18_v1b"
num_classes = 3
learning_rate = 1e-3
model = TimmLightningModel(model_name=model_name,num_classes=num_classes,learning_rate=learning_rate)

saved_state_dict = torch.load(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda")
model.load_state_dict(saved_state_dict)
# model.to(device)
model.eval()

threshold = 0.2
app = FastAPI()

class ArrayRequest(BaseModel):
    arr: str

@app.post("/eval/")
async def process_image(arr: ArrayRequest):
    # better than pytorch way
    arr = np.array(json.loads(arr.arr))
    # print(arr)
    arr = preprocess_image(arr)
    prob = model.predict(arr)
    print(prob)
    prob = (prob > threshold).int()
    prob = "".join(prob[0].numpy().astype(str))
    
    # list_data_prob = []
    # for data_arr in arr:
    #     data_arr = preprocess_image(data_arr)
    #     prob = model.predict(data_arr)
    #     prob = (prob > threshold).int()
    #     prob = "".join(prob[0].numpy().astype(str))
    #     list_data_prob.append(prob)

    # print(len(list_data_prob))
    return PlainTextResponse(prob)

if __name__ == "__main__":
    uvicorn.run("frb_fastapi:app", port=8911, log_level="info")