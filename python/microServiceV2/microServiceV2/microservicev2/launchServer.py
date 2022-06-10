import sys
import struct
import torch
import pickle
import os
import torchvision

from http.server import BaseHTTPRequestHandler, HTTPServer

from microservicev2.decodeHeif import convertBytes
from microservicev2.runModel import process, runPipeline

BACK_DETECTOR_MODEL=None
YOLO_MODEL, LIGHT_YOLO_MODEL=None, None
CNN_SVM, SVM_CLASSIFIER=None, None
YOLO, YOLO_LIGHT=None, None

class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        global BACK_DETECTOR_MODEL, CNN_SVM, SVM_CLASSIFIER, YOLO, YOLO_LIGHT

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        content_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(content_len)

        img = convertBytes(content)

        verdict = runPipeline(img, BACK_DETECTOR_MODEL, CNN_SVM, SVM_CLASSIFIER, YOLO, YOLO_LIGHT)

        self.wfile.write(struct.pack('>f', verdict))

    def log_message(self, format, *args):
        return

def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

def launchServer(host, port, modelpath):
    global CNN_SVM, SVM_CLASSIFIER, BACK_DETECTOR_MODEL, CNN_SVM, SVM_CLASSIFIER, YOLO, YOLO_LIGHT
    with open(f'{modelpath}/SVMCNN.pckl', 'rb') as handle:
        CNN_SVM = pickle.load(handle)
    with open(f'{modelpath}/SVMclf.pckl', 'rb') as handle:
        SVM_CLASSIFIER = pickle.load(handle)
    BACK_DETECTOR_MODEL = load_model(f'{modelpath}/backDetector.pt').cpu()
    YOLO = torch.hub.load('azazazazazazazaza/yolov5', 'custom', path=f"{modelpath}/YOLO.pt").cpu()
    with open(f'{modelpath}/light.pckl', 'rb') as handle:
        YOLO_LIGHT = pickle.load(handle)

    webServer = HTTPServer((host, port), Server)
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        webServer.server_close()