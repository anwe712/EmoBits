from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from emotion_detector import detect_emotion
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect_emotion", methods=["POST"])
def process_frame():
    file = request.files["frame"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    detected_emotion = detect_emotion(img)
    
    return jsonify({"emotion": detected_emotion})

if __name__ == "__main__":
    app.run(debug=True)
