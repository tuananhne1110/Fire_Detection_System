from flask import Flask, render_template, request, redirect, url_for, session,jsonify, send_file, Response
import ffmpeg
import subprocess

# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_session import Session
# from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import io
from PIL import Image
import base64
import os
import sqlite3
import cv2 as cv
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash
from function import predict_image,convert_video_for_web
from gridfs import GridFS
import db_functions as dbf

# Generate a random secret key
secret_key = os.urandom(24)
app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['SESSION_TYPE'] = 'filesystem'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
url = "rtsp://admin:Ditmemay@192.168.52.220/onvif1"
# login_manager = LoginManager()
# login_manager.init_app(app)

Session(app)

# class User(UserMixin):
#     def __init__(self, user_id):
#         self.id = user_id
#         self.name = None  # Add more user-specific attributes if needed

# @login_manager.user_loader
# def load_user(user_id):
#     # Retrieve the user from the database based on user_id
#     user = collection.find_one({"username": user_id})
#     if user:
#         return User(user['username'])
#     return None
#
#
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["mydatabase"]
# collection = db["tkmk"]
# data = {
#     "username": "anh",
#     "password": "anh"
# }
#
# def check_credentials(username, password):
#     user = collection.find_one({"username": username, "password": password})
#
#     if user:
#         return True
#     else:
#         collection.insert_one(data)
#         return False
# check_credentials(data['username'],data['password'])



@app.route('/')
@app.route('/home')
def home():
    return render_template('base.html')


@app.route('/demo')
def index():
    return render_template('index.html')


@app.route('/signin/', methods=['GET', 'POST'])
def signin():
#     session.clear()
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')
#         user = collection.find_one({"username": username})

#         if user and check_password_hash(user['password'], password):
#             session['username'] = username  
#             print('session',session['username'])
#             return render_template('camera.html', name=session['username'])

#         else:
#             return "Incorrect username or password", 401

    return render_template('signin.html')
# @app.route('/signin/', methods=['GET', 'POST'])
# # def checkSession(session):
# #     if session == None:
# #         print('Session no clear r m')
# #         return True
# #     print('session no ne',session)
    
# # checkSession(session)
# def signin():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')
#         user = collection.find_one({"username": username})  # Query your database for the user
#         print('Username:', username)
#         if user and user.get('password') == password:  # Check if the user exists and the password matches
#             user_obj = User(user['username'])
#             login_user(user_obj)
#             session['username'] = user['username']  # Set the username in the session

#             return redirect(url_for('camera'))  # Redirect to a protected page

#         else:
#             return "Incorrect username or password", 401

#     return render_template('signin.html')



@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # image_blob = request.files['image_blob']
        image = request.files['data']
        if image:
            print('iamge_file_name',filename)
            print('image_filename',image.filename)
            print('image',image)
            filename = os.path.join(os.path.dirname("."), "uploads", image.filename)
            image.save(filename)

            return jsonify({ "message": "luu thanh cong roi nek" })
        else:
            return jsonify({"error": "No image blob received"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/camera')
# @login_required
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

@app.route('/process-video', methods=['POST'])
def testcam():
    try:
        image_blob = request.files['image_blob']
        print(image_blob)

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
        print(e)
        return jsonify({"error": str(e)}), 500
# 2/11 - get video

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload-video', methods=['POST'])
# def upload_video():
#     if 'data' not in request.files:
#         return jsonify({'error': 'No video part'})

#     video = request.files['data']

#     if video.filename == '':
#         return jsonify({'error': 'No selected video file'})

#     if video and allowed_file(video.filename):
#         client = pymongo.MongoClient("mongodb://localhost:27017")
#         db = client["gridfs_media"]
#         fs = GridFS(db)

#         with video.stream as file_stream:
#             file_id = fs.put(file_stream, filename=video.filename)

#         print(f"File stored in GridFS with ID: {file_id}")

#         cap = cv2.VideoCapture(file_stream)

#         if not cap.isOpened():
#             return jsonify({'error': 'Failed to open the uploaded video file'})

#         processed_frames = []  # This will hold the processed frames
#         frame_list = []

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Use your predict_image function to process the frame
#             processed_frame = predict_image(frame)

#             frame_filename = f"frame_{len(frame_list)}.jpg"
#             frame_path = os.path.join(os.path.dirname("."), "frames", frame_filename)
#             cv2.imwrite(frame_path, processed_frame)

#             frame_list.append(frame_filename)
#             processed_frames.append(processed_frame)

#         cap.release()

#         # Optional: Save the processed frames as a new video
#         output_path = os.path.join(os.path.dirname("."), "uploads", "processed_" + video.filename)
#         out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
#         for pf in processed_frames:
#             out.write(pf)
#         out.release()
#         processed_path = os.path.join(os.path.dirname("."), "static", "processed_" + video.filename)

#         convert_video_for_web(output_path, processed_path)

#         # 'frames': frame_list,
#         return jsonify({'message': 'Video uploaded and processed successfully', 'filename': processed_path})

#     return jsonify({'error': 'Invalid file format'})
def upload_video():
    if 'data' not in request.files:
        return jsonify({'error': 'No video part'})

    video = request.files['data']

    if video.filename == '':
        return jsonify({'error': 'No selected video file'})

    if video and allowed_file(video.filename):
        filename = os.path.join(os.path.dirname("."), "static", video.filename)
        video.save(filename)
        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            return jsonify({'error': 'Failed to open the uploaded video file'})

        processed_frames = []  # This will hold the processed frames
        frame_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Use your predict_image function to process the frame
            processed_frame = predict_image(frame)

            frame_filename = f"frame_{len(frame_list)}.jpg"
            frame_path = os.path.join(os.path.dirname("."), "frames", frame_filename)
            cv2.imwrite(frame_path, processed_frame)

            frame_list.append(frame_filename)
            processed_frames.append(processed_frame)

        cap.release()

        # Optional: Save the processed frames as a new video
        output_path = os.path.join(os.path.dirname("."), "uploads", "processed_" + video.filename)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
        for pf in processed_frames:
            out.write(pf)
        out.release()
        processed_path = os.path.join(os.path.dirname("."), "static", "processed_" + video.filename)
        print("du me",  output_path, processed_path)
        ffmpeg_executable = "C:\\Users\\CONG DUY\\Downloads\\ffmpeg-2023-11-05-git-44a0148fad-full_build\\ffmpeg-2023-11-05-git-44a0148fad-full_build\\bin\\ffmpeg.exe"  # Provide the full path to ffmpeg.exe

        convert_video_for_web(output_path, processed_path, ffmpeg_executable)

        # convert_video_for_web(output_path, processed_path)

        # 'frames': frame_list,
        dbf.store_and_display_media(processed_path, "GRIDFS_DBB", processed_path)
        return jsonify({'message': 'Video uploaded and processed successfully', 'filename': processed_path})

    return jsonify({'error': 'Invalid file format'})

def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mkv'}  
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    camera_source = session.get('camera_source', 'webcam')
    if camera_source == 'ip_camera':
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)