import sys
import os
import torch
from PIL import Image
import numpy as np
import torchvision

MODEL_PATH = "./basic.pt"

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

def downsize(inp_path, out_path):
    image = Image.open(inp_path)
    if image.size == (4032, 3024):
        image = image.rotate(270, expand=True)
    image.thumbnail((75, 100), Image.ANTIALIAS)
    image.save(out_path)

def runModel(path, model):
    os.system(f".\Heic2jpg.exe {path} tmp.jpg")
    downsize("./tmp.jpg", "prep.jpg")
    image = Image.open("prep.jpg")
    tr = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = torch.reshape(tr(image), (1, 3, 100, 75))
    return float(model(x.to("cuda:0"))[0][0])

