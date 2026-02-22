from flask import Flask, request, send_file
import cv2
import os
import requests

app = Flask(__name__)

# Model Download (EDSR - Best for 4K)
MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
MODEL_PATH = "EDSR_x4.pb"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... Please wait.")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded!")

ensure_model()
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("edsr", 4)

@app.route('/upscale', methods=['GET'])
def upscale():
    image_url = request.args.get('url')
    if not image_url: return {"error": "URL missing"}, 400

    img_path = "input.jpg"
    out_path = "output.png"
    
    try:
        # Download
        r = requests.get(image_url, timeout=15)
        with open(img_path, 'wb') as f: f.write(r.content)

        # Read & Process
        img = cv2.imread(img_path)
        # RAM bachanay ke liye agar image bari hai toh optimize karein
        h, w = img.shape[:2]
        if h > 800 or w > 800:
            img = cv2.resize(img, (w//2, h//2))

        result = sr.upsample(img)
        cv2.imwrite(out_path, result)
        
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(img_path): os.remove(img_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
