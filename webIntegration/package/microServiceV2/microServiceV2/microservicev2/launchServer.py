import sys
import struct
import torch
import pickle
import os
import torchvision
import sqlite3
import base64
import io

from http.server import BaseHTTPRequestHandler, HTTPServer

from microservicev2.decodeHeif import convertBytes
from microservicev2.runModel import process, runPipeline

BACK_DETECTOR_MODEL=None
YOLO_MODEL, LIGHT_YOLO_MODEL=None, None
CNN_SVM, SVM_CLASSIFIER=None, None
YOLO, YOLO_LIGHT=None, None
DB_CON, DB_CUR = None, None

class Server(BaseHTTPRequestHandler):
    def do_POST(self):
        global BACK_DETECTOR_MODEL, CNN_SVM, SVM_CLASSIFIER, YOLO, YOLO_LIGHT, DB_CON, DB_CUR
        print("POST")

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        content_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(content_len)

        header, encoded = content.split(b',', 1)
        byteStringImg = base64.b64decode(encoded)
        byteIO = io.BytesIO(byteStringImg)
        byteIO.seek(0)
        byteIMG = byteIO.read()
        sql = f"INSERT INTO images (img) VALUES (?)"
        DB_CON.execute(sql, [sqlite3.Binary(byteIMG)] )
        DB_CON.commit()

        img = convertBytes(content)

        verdict = runPipeline(img, BACK_DETECTOR_MODEL, CNN_SVM, SVM_CLASSIFIER, YOLO, YOLO_LIGHT)
        
        print(verdict)
        self.wfile.write(bytes(str(verdict), "utf-8"))

    def log_message(self, format, *args):
        return

def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

def launchServer(host, port, modelpath, db_path):
    global CNN_SVM, SVM_CLASSIFIER, BACK_DETECTOR_MODEL, CNN_SVM, SVM_CLASSIFIER, YOLO, YOLO_LIGHT, DB_CON, DB_CUR
    with open(f'{modelpath}/SVMCNN.pckl', 'rb') as handle:
        CNN_SVM = pickle.load(handle)
    with open(f'{modelpath}/SVMclf.pckl', 'rb') as handle:
        SVM_CLASSIFIER = pickle.load(handle)
    BACK_DETECTOR_MODEL = load_model(f'{modelpath}/backDetector.pt').cpu()
    YOLO = torch.hub.load('azazazazazazazaza/yolov5', 'custom', path=f"{modelpath}/YOLO.pt").cpu()
    with open(f'{modelpath}/light.pckl', 'rb') as handle:
        YOLO_LIGHT = pickle.load(handle)
    
    DB_CON = sqlite3.connect(db_path)
    DB_CUR = DB_CON.cursor()

    webServer = HTTPServer((host, port), Server)
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        webServer.server_close()