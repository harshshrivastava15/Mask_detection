import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)
# h

# Load the pre-trained model
model = load_model('model.h5')

def detect_and_crop_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return [(frame[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]

def detect_mask_in_frame(face_crop, model, Image_size=100):
    resized_face = cv2.resize(face_crop, (Image_size, Image_size))
    resized_face = np.expand_dims(resized_face / 255.0, axis=0)
    prediction = model.predict(resized_face)
    predicted_class = ['Masked', 'Not Masked'][np.argmax(prediction)]
    return predicted_class, prediction[0][np.argmax(prediction)]

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        detected_faces = detect_and_crop_faces(frame)

        for face_crop, (x, y, w, h) in detected_faces:
            predicted_class, confidence = detect_mask_in_frame(face_crop, model)
            color = (0, 255, 0) if predicted_class == 'Masked' else (0, 0, 255)
            label = f"{predicted_class}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

