from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import base64
import convert
import struct
import torch

MODEL_PATH = "./basic.pt"

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

MODEL = load_model(MODEL_PATH)

class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        global model 
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        content_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(content_len)
        img = base64.b64decode(body)
        with open("REP.HEIC", "wb") as img_file:
            img_file.write(img)
        res = convert.runModel(".\REP.HEIC", MODEL)
        self.wfile.write(struct.pack('>f', res))
		
    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    hostName, serverPort = sys.argv[1], int(sys.argv[2])
    print(hostName, serverPort)
    webServer = HTTPServer((hostName, serverPort), Server)
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        webServer.server_close()