import requests
import base64
import struct
import sys
import socket

from PIL import Image


hostname = socket.gethostname()    
IPAddr = socket.gethostbyname(hostname)   
url = f"http://{IPAddr}:{8888}"
img = base64.b64encode(open("data.jpg", "rb").read())
response = requests.get(url, data=img)
transTable = {0: "Здоровые", 1:"Белая гниль", 2:"Бурая ржавчина"}
print(transTable[struct.unpack('>f', response.content)[0]])
input()