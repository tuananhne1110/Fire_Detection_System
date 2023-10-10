from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import io
from PIL import Image
import base64

app = Flask(__name__)

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
    #     username = request.form['username']
    #     password = request.form['password']
    #     fullname = request.form['fullname']
    #     email = request.form['email']   
    return render_template('signin.html')


@app.route('/process-image', methods=['POST'])
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
    
 
if __name__ == '__main__':
    app.run(debug=True)
#4:00 10/10