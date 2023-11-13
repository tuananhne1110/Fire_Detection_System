import math
from ultralytics import YOLO
import cv2
import os
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import ffmpeg
import numpy as np
from datetime import datetime
import shutil
import threading
import queue
from threading import Timer


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


class ImageProcessor:
    def __init__(self, file):
        self.file = file

    def process(self):
        print(' process image is running')
        try:

            # Read image file and convert it to numpy array
            image_np = np.frombuffer(self.file.read(), np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Predict Image
            g_image, is_detected = predict_image(image)

            return g_image, is_detected
        except Exception as e:
            print(e)
            print('Error with type of input')
            return None, False


class VideoProcessor:
    print('Video processor is running')

    def __init__(self, filename):
        self.filename = filename

    def process(self):
        try:
            cap = cv2.VideoCapture(self.filename)
            processed_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frames using predict_image
                processed_frame, detected = predict_image(frame)

                processed_frames.append(processed_frame)

            cap.release()

            # Save video temporarily
            temp_output_path = os.path.join(os.path.dirname("."), "uploads", "temp_" + os.path.basename(self.filename))
            out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
            for pf in processed_frames:
                out.write(pf)
            out.release()

            # Convert videos to optimize for the web
            final_output_path = os.path.join(os.path.dirname("."),
                                             "static", "processed_" + os.path.basename(self.filename))
            convert_video_for_web(temp_output_path, final_output_path)

            return final_output_path
        except Exception as e:
            print(e)
            return None


def send_warning_email(image_with_bboxes):
    print('sending_warning_email is running')
    # Save the image temporarily to attach it in the email
    temp_image_path = os.path.join(FOLDER_NOW_PATH, "temp_alert.jpg")
    cv2.imwrite(temp_image_path, image_with_bboxes)

    msg = MIMEMultipart()
    msg['From'] = email_from
    msg['Subject'] = subject
    body = f"This is an automated warning about a potential fire detected in the surveillance area {formatted_time}."
    msg.attach(MIMEText(body, 'plain'))

    with open(temp_image_path, 'rb') as attachment:
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload(attachment.read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition',
                                      f'attachment; filename={os.path.basename(temp_image_path)}')
        msg.attach(attachment_package)

    text = msg.as_string()

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_from, pswd)
        for email_to in email_list:
            msg['To'] = email_to
            server.sendmail(email_from, email_to, text)

    # Remove the temporary image after sending
    os.remove(temp_image_path)
    print('passed sending_warning_email')


last_frame = None


def update_last_frame():
    global last_frame
    q = queue.Queue()
    url = 'rtsp://admin:Ditmemay1@192.168.1.173:554/onvif1'
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        last_frame = frame
        try:
            q.get_nowait()  # discard previous (unprocessed) frame
        except queue.Empty:
            pass
        q.put(frame)


t = threading.Thread(target=update_last_frame)
t.daemon = True
t.start()


def reset_email_flag():  # reset flag =# to send email after 5 minutes
    global email_flag
    email_flag = 0


def predict_image(image_input):
    detected = 0  # Flag to determine if a bounding box was detected

    # Define new frame dimensions
    new_width, new_height = 640, 480

    # Resize the image
    img = cv2.resize(image_input, (new_width, new_height))

    # Get results from the YOLO model
    results = model(img, stream=True)

    # Draw bounding boxes and confidence scores on detected fires
    for info in results:
        boxes = info.boxes

        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 30:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(img, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
                detected = 1  # Set the flag to True if a bounding box was detected

    return img, detected


def gen_frames():  # generate frame by frame from RTSP stream
    global last_frame
    global email_flag
    print('gen_frames is running')

    while True:
        # success, frame = cap.read()  # read the camera frame
        frame = last_frame
        success = True
        if not success:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("Failed to grab frame")
                break
        else:
            # is_detected use for sending warming email, when is_detected = 1 then send email
            frame, is_detected = predict_image(frame)
            if is_detected == 1 and email_flag == 0:  # If a bounding box was detected
                print('prepare sending email')
                send_warning_email(frame)
                email_flag = 1 
                timer = Timer(300, reset_email_flag)
                timer.start
                print(" passed sending email")
            # else:
            #     print( 'con cac')            

            # Encode the frame with bounding boxes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concat frame one by one and show result

    cap.release()
    print(' passed gen_frames')


def gen_frames_webcam():  # generate frame by frame from laptop webcam
    print(' gen_frames_webcam is running')
    global email_flag
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    if not cap.isOpened():
        raise ValueError("Error: Could not open webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break  # If error in capturing frame, break loop

        frame, is_detected = predict_image(frame)  # Process frame for bounding box
        if is_detected == 1 and email_flag == 0:  # If a bounding box was detected
            send_warning_email(frame)
            email_flag = 1 
            timer = Timer(300, reset_email_flag)
            timer.start
            print(" passed sending email")

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # If error in encoding frame, continue to the next frame

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("passed gen_frames_webcam")


ffmpeg_path = shutil.which('ffmpeg')


def convert_video_for_web(input_path, output_path):
    print('convert_video_for_web is running')
    try:
        # Convert videos with H.264 codec and package in MP4 container.
        # Add fake audio if video has no audio.
        # Use 'faststart' to move the MOOV atom to the beginning of the file (optimal for streaming).
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec='libx264', movflags='faststart', acodec='aac', strict='experimental')
            .run(overwrite_output=True, cmd=ffmpeg_path)
        )
        print('Passed convert_video_for_web')
    except ffmpeg.Error as e:
        print(f"Error when convert video: {e}")

