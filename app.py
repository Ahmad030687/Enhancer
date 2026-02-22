from flask import Flask, request, send_file
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)

def enhance_image(img, scale_factor):
    # 1. Upscale using Lanczos4 (Best for high-quality resizing)
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    upscaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # 2. Sharpening Filter (Mimics AI detail recovery)
    # Ye pixels ko crisp banata hai taake 4K feel aaye
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    # 3. Denoising (Slight blur to remove artifacts)
    final = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
    return final

@app.route('/upscale', methods=['GET'])
def upscale():
    url = request.args.get('url')
    mode = request.args.get('mode', '4k') # default 4k
    
    if not url: return {"error": "URL kahan hai?"}, 400

    img_path = "temp_in.jpg"
    out_path = "temp_out.png"

    try:
        # Download
        r = requests.get(url, timeout=15)
        with open(img_path, 'wb') as f: f.write(r.content)

        img = cv2.imread(img_path)
        
        # Mode Selection
        # 4K = ~4x scale, 8K = ~8x scale
        scale = 4 if mode == '4k' else 8
        
        # Processing
        result = enhance_image(img, scale)
        
        cv2.imwrite(out_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return send_file(out_path, mimetype='image/png')

    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(img_path): os.remove(img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    
