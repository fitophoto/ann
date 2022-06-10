import pillow_heif
import io
import base64

from PIL import Image

def convertBytes(byteString):
    try:
        byteStringImg = base64.b64decode(byteString)
        byteIO = io.BytesIO(byteStringImg)
        byteIO.seek(0)
        byteIMG = byteIO.read()
        heif_file = pillow_heif.read_heif(byteIMG)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
    except:
        byteStringImg = base64.b64decode(byteString)
        byteIO = io.BytesIO(byteStringImg)
        image = Image.open(byteIO).resize((400, 400))
    return image