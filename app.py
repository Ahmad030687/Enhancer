from flask import Flask, request, send_file
import cv2
import numpy as np
import requests
import os

app = Flask(__name__)

# Function to enhance image quality (4K/8K Hybrid logic)
def enhance_image(img, scale_factor):
    # 1. High Quality Upscaling using Lanczos4
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    
    # RAM Guard: Agar result bohot bada ho raha ho toh limit lagana (for 512MB RAM)
    if width > 6000 or height > 6000:
        scale_factor = 2
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)

    upscaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # 2. Sharpening (Pixels ko crisp karne ke liye)
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    # 3. Final Polish (Color and detail refinement)
    final = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
    return final

@app.route('/')
def home():
    return "<h1>Ahmad RDX 8K Enhancer API is Live!</h1>"

@app.route('/upscale', methods=['GET'])
def upscale():
    url = request.args.get('url')
    mode = request.args.get('mode', '4k')
    
    if not url:
        return {"error": "URL missing! Please provide an image URL."}, 400

    img_path = "temp_in.jpg"
    out_path = "temp_out.png"

    try:
        # Download image with Headers (To bypass bot blocks)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(url, headers=headers, timeout=15)
        
        if r.status_code != 200:
            return {"error": f"Failed to download image. Status: {r.status_code}"}, 400
            
        with open(img_path, 'wb') as f:
            f.write(r.content)

        # Load image into OpenCV
        img = cv2.imread(img_path)
        
        # Check if image was correctly loaded (Fixes the 'NoneType' error)
        if img is None:
            return {"error": "Could not read image. Please use a direct image link (.jpg, .png)"}, 400

        # Select Scaling Factor
        # 4k mode = 4x scale, 8k mode = 8x scale
        scale = 3 if mode == '4k' else 6 # 512MB RAM ke liye thoda optimize rakha hai
        
        # Process image
        result = enhance_image(img, scale)
        
        # Save result
        cv2.imwrite(out_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        
        return send_file(out_path, mimetype='image/png')

    except Exception as e:
        return {"error": str(e)}, 500
    
    finally:
        # Cleanup: Remove files after sending to save server space
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(out_path): 
            # File sending ke baad remove karne ke liye delay ya response handling chahiye hoti hai, 
            # filhal temp save hone dein ya restart par delete.
            pass

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
