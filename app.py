from flask import Flask, request, send_file
import cv2
import os
import requests
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import gc

app = Flask(__name__)

# Model ko CPU-Only aur Memory Efficient rakha hai
def get_upsampler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # tile=128 matlab image ko chote tukron mein process karega
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=128, 
        tile_pad=10,
        pre_pad=0,
        half=False # CPU mode
    )
    return upsampler

upsampler = get_upsampler()

@app.route('/upscale', methods=['GET'])
def upscale():
    image_url = request.args.get('url')
    if not image_url: return {"error": "URL do jaani!"}, 400

    img_path = "input.jpg"
    out_path = "output_4k.png"
    
    try:
        # 1. Download
        r = requests.get(image_url, timeout=10)
        with open(img_path, 'wb') as f:
            f.write(r.content)

        # 2. Load Image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # 3. RAM Guard: Agar image bohot bari hai toh thoda compress karo
        h, w = img.shape[:2]
        if h * w > 1000000: # 1MP se bari image
            img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)

        # 4. Processing (4K Max Mode)
        output, _ = upsampler.enhance(img, out_scale=4)
        
        # 5. Save & Cleanup
        cv2.imwrite(out_path, output)
        
        # Memory saaf karna zaroori hai
        del output
        gc.collect() 
        
        return send_file(out_path, mimetype='image/png')
    
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(img_path): os.remove(img_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
