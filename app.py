from flask import Flask, request, send_file
import cv2
import os
import requests
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch

app = Flask(__name__)

# Model Setup (Real-ESRGAN)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False
)

@app.route('/upscale', methods=['GET'])
def upscale():
    image_url = request.args.get('url')
    if not image_url:
        return {"error": "URL missing"}, 400

    # Image download karein
    img_path = "input.jpg"
    out_path = "output_8k.png"
    
    r = requests.get(image_url)
    with open(img_path, 'wb') as f:
        f.write(r.content)

    # AI Processing
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, out_scale=4) # out_scale=4 se 4k/8k banta hai

    cv2.imwrite(out_path, output)
    return send_file(out_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
  
