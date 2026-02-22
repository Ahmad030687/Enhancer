from flask import Flask, request, send_file
import cv2
import os
import requests

app = Flask(__name__)

@app.route('/upscale', methods=['GET'])
def upscale():
    image_url = request.args.get('url')
    if not image_url: return {"error": "URL do bhai!"}, 400

    img_path = "in.jpg"
    out_path = "out_4k.png"
    
    try:
        # Download
        r = requests.get(image_url, timeout=10)
        with open(img_path, 'wb') as f: f.write(r.content)

        # Image Load
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 4K Calculation (Width 3840 tak le jana)
        target_width = 3840
        ratio = target_width / float(w)
        target_height = int(h * ratio)

        # Ultra HD Upscaling (Lanczos4 is best alternative to AI)
        result = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Details Sharpness (Heavy factor)
        # Isse image AI enhancer jaisi crisp ho jati hai
        gaussian = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)

        cv2.imwrite(out_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(img_path): os.remove(img_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
