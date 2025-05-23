from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load face detection and recognition model
model_path = os.path.join(os.getcwd(), 'models', 'openface.nn4.small2.v1.t7')
face_model = cv2.dnn.readNetFromTorch(model_path)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    face_model.setInput(face_blob)
    vec = face_model.forward()
    return vec.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    uploaded_file = request.files['image']
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reference.jpg')
    uploaded_file.save(ref_path)

    ref_embedding = get_face_embedding(ref_path)
    if ref_embedding is None:
        return render_template("result.html", result="❌ No face found in uploaded image.", ref=None, live=None)

    # Capture live image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    test_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live.jpg')
    cv2.imwrite(test_path, frame)

    test_embedding = get_face_embedding(test_path)
    if test_embedding is None:
        return render_template("result.html", result="❌ No face found in webcam image.", ref='uploads/reference.jpg', live=None)

    # Compare using cosine similarity
    similarity = np.dot(ref_embedding, test_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(test_embedding))
    result = "✅ Face Matched" if similarity > 0.5 else "❌ Not Matched"

    return render_template('result.html',
                           result=result,
                           ref='uploads/reference.jpg',
                           live='uploads/live.jpg')

if __name__ == '__main__':
    app.run(debug=True)
