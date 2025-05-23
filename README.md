:

🧠 Face Verification Web App Using Python & OpenCV
📌 Project Description
This is a Python-powered web application that performs real-time face verification using Flask for the web interface and OpenCV’s deep learning module for facial feature extraction and comparison.

The app allows users to:

Upload a reference photo

Capture a live image using the webcam

Detect and compare both faces

Display a match result: ✅ Face Matched or ❌ Not Matched

Visually present both the uploaded and live-captured images on the result screen

This project highlights Python’s capabilities in computer vision, real-time camera integration, and lightweight web development using Flask.

🧰 Core Technologies
💻 Backend:
Python 3.12+

Flask: For building the web routes and rendering templates

OpenCV:

Haar Cascades: For face detection

DNN Module: To load the OpenFace pre-trained deep learning model (openface.nn4.small2.v1.t7)

NumPy: For vectorized cosine similarity calculation

🌐 Frontend:
HTML5 with Bootstrap for minimal styling and responsive design

Jinja2 (Flask template engine) for dynamic content rendering

🚀 Key Features
Upload a face image via browser

Live webcam capture using OpenCV

Deep feature extraction using OpenFace model

Face comparison via cosine similarity

Display uploaded and captured images on result page

User-friendly error messages when no face is detected

📂 Project Workflow
User visits the homepage and uploads a face image.

Flask saves the image and calls OpenCV to extract a facial embedding.

Simultaneously, the app activates the webcam, captures a live image, and extracts its facial features.

Python compares both embeddings using cosine similarity.

The result is displayed along with both images for visual confirmation.

✅ Use Cases
Real-time identity verification

Face login systems

Face-based access control in web interfaces

Prototyping computer vision applications



