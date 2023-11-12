from flask import Flask, render_template, request, jsonify, Response
from flask_session import Session
import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from datetime import datetime
from build_function import ImageProcessor, VideoProcessor, gen_frames, gen_frames_webcam
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'


Session(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Email Configuration
email_flag = 0
smtp_port = 587
smtp_server = "smtp.gmail.com"
email_from = "truongnnse173216@fpt.edu.vn"
email_list = ["masayukibalad@gmail.com"]
pswd = 'fzkg nyoa oeaa wyis'
subject = "WARNING"
formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Path
FOLDER_NOW_PATH = './uploads'
model = YOLO("./best.pt")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
url = 'rtsp://admin:Ditmemay1@192.168.1.173:554/onvif1'


@app.route('/')
@app.route('/home')
def home():
    return render_template('base.html')


@app.route('/demo')
def index():
    return render_template('index.html')


@app.route('/process-image', methods=['POST'])
def process_image():
    print('process_image is running')
    global email_flag
    try:
        image = request.files['data']
        if image:
            # Save the original image temporarily
            original_filename = os.path.join(os.path.dirname("."), "static", image.filename)
            image.save(original_filename)

            # Process images via ImageProcessor
            with open(original_filename, 'rb') as file:
                image_processor = ImageProcessor(file)
                g_image, is_detected = image_processor.process()

            # Save the processed image
            processed_filename = "processed_" + image.filename
            processed_filepath = os.path.join(os.path.dirname("."), "static", processed_filename)
            cv2.imwrite(processed_filepath, g_image)
            print('passed process-image')
            return jsonify({"message": "Image processed and saved successfully", 'filename': processed_filepath})
        
        else:
            return jsonify({"error": "No image blob received"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/process-video', methods=['POST'])
def testcam():
    print('process video is running')
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
            print('passed process video')
            return jsonify({"gray_image": gray_image_base64})
        else:
            return jsonify({"error": "No image blob received"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/upload-video', methods=['POST'])
def upload_video():

    print(" upload video is running")
    if 'data' not in request.files:
        return jsonify({'error': 'There is no video section'})

    video = request.files['data']

    if video.filename == '':
        return jsonify({'error': 'The video file is not selected'})

    if video and allowed_file(video.filename):
        filename = os.path.join(os.path.dirname("."), "static", video.filename)
        video.save(filename)
        print('saved video')

        video_processor = VideoProcessor(filename)
        processed_filename = video_processor.process()
        print('Passed upload_video')
        return jsonify({'message': 'The video has been uploaded and processed successfully', 'filename': processed_filename})

    return jsonify({'error': 'Invalid file format'})


def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_feed')
def webcam_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
