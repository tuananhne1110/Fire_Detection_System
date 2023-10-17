from flask import Flask, render_template, request, redirect, url_for, jsonify, Response,session
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import io
from PIL import Image
import base64
import os

# Generate a random secret key
secret_key = os.urandom(24)
app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)


@app.route('/')
@app.route('/home')
def home():
    return render_template('base.html')

@app.route('/demo')
def index():
    return render_template('index.html')


@app.route('/signin', methods = ['GET','POST'])
def signin():
    # if request.method == 'POST':
    #     # Xử lý đăng nhập ở đây
    username = request.form.get('username')
    #     password = request.form['password']
    #     fullname = request.form['fullname']
    #     email = request.form['email']   
    session['username'] = username
    return render_template('signin.html', username=username)


@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        image_blob = request.files['image_blob']
        if image_blob:
            # Read the image blob as bytes
            image_data = image_blob.read()

            # Convert the image data to a NumPy array
            image_np = np.frombuffer(image_data, np.uint8)

            # Decode the image using OpenCV
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Encode the grayscale image to a base64 string
            _, buffer = cv2.imencode('.jpg', gray_image)
            gray_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Return the grayscale image as a base64 string
            return jsonify({"gray_image": gray_image_base64})
        else:
            return jsonify({"error": "No image blob received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/camera')
def camera():
    return render_template('camera.html') 

@app.route('/save_video', methods=['POST'])
def save_video():
    try:
        video_blob = request.files['video_blob']
        if video_blob:
            # Save the video blob to a file on the server
            video_blob.save('saved_video.webm')

            # Return a response to the client
            return jsonify({"message": "Video saved successfully"})
        else:
            return jsonify({"error": "No video blob received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
   
@app.route('/get_video', methods=['GET'])
def get_video():
    try:
        # Specify the path to the saved video file
        video_path = 'videos/saved_video.webm'

        # Send the video file to the client for download
        return send_file(video_path, mimetype='video/webm', as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded image to the UPLOAD_FOLDER
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Process the image to grayscale
        img = Image.open(filename).convert('L')
        processed_filename = os.path.join(app.config['PROCESSED_FOLDER'], 'grayscale_' + file.filename)
        img.save(processed_filename)

        return render_template('index.html', original_image=file.filename, grayscale_image='grayscale_' + file.filename)
 
if __name__ == '__main__':
    app.run(debug=True)
