from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
model = load_model('model.h5')

def detect_and_crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return [(image[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]

def detect_mask_in_frame(face_crop, model, Image_size=100):
    resized_face = cv2.resize(face_crop, (Image_size, Image_size))
    resized_face = np.expand_dims(resized_face / 255.0, axis=0)
    prediction = model.predict(resized_face)
    predicted_class = ['Masked', 'Not Masked'][np.argmax(prediction)]
    return predicted_class, float(prediction[0][np.argmax(prediction)])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image data from the request
    image_data = request.json['image']
    
    # Decode the base64 image
    image_data = image_data.split(',')[1]
    image_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Detect faces and process each face
    detected_faces = detect_and_crop_faces(image)
    results = []
    
    for face_crop, (x, y, w, h) in detected_faces:
        predicted_class, confidence = detect_mask_in_frame(face_crop, model)
        results.append({
            'class': predicted_class,
            'confidence': confidence,
            'bbox': [x, y, w, h]
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)