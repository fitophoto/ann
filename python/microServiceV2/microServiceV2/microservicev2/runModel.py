from PIL import Image
import torch
import torchvision
import os
import numpy as np

def process(img):
    img = img.resize((400,400))
    tr = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return torch.reshape(tr(img), (1, 3, 400, 400))

"""
0-здоровые 
1-гниль 
2-ржавична
"""
def runPipeline(img, bckgClassifier, cnnSvm, svm, yolo, light):
    proc_img = process(img)
    pred = float(bckgClassifier(proc_img)[0][0])

    verdict = None
    if pred > 0.5:
        img = img.resize((416, 416))
        detections = yolo([img])
        preds = detections.pred

        for i in range(len(detections.pred)):
            detections.pred[i] = detections.pred[i][detections.pred[i][:, 4] > 0.4]
        
        detections.save(labels=False, save_dir=os.getcwd())

        max_len = 50

        preds[0] = preds[0][:, 4:]
        conf = preds[0][:, 0]
        argsort = torch.argsort(conf, dim=0, descending=True)
        preds[0] = preds[0].flatten()[: max_len * 2].tolist()
        preds[0] += [0.] * (max_len * 2 - len(preds[0])) + [len(preds[0])]
        preds[0] = torch.tensor(preds[0], dtype=torch.float32)
        with torch.no_grad():
            verdict = int(np.argmax(light(preds[0].cpu()).cpu()))
        transTable = {0: 0, 1: 2, 2: 1}
        verdict = transTable[verdict]
    else:
        features = cnnSvm(proc_img).detach().numpy()
        verdict = svm.predict(features)[0]

    return verdict