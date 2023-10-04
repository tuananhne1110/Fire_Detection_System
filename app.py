from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import io
from PIL import Image
import base64
import random
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        uploaded_image = request.files['image']
        if uploaded_image:
            image_data = uploaded_image.read()

            # Convert the image data to a NumPy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Generate random starting position (x, y) inside the image
            image_height, image_width, _ = image.shape
            square_size = 50
            x = (image_width - square_size) // 2
            y = (image_height - square_size) // 2

            # Draw a random square inside the image
            cv2.rectangle(image, (x, y), (x + square_size, y + square_size), (0, 0, 255), 2)

            # Convert to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save grayscale image to memory
            grayscale_buffer = io.BytesIO()
            grayscale_pil_image = Image.fromarray(grayscale_image)
            grayscale_pil_image.save(grayscale_buffer, format="PNG")
            grayscale_data = grayscale_buffer.getvalue()

            # Encode the grayscale image to base64
            grayscale_base64 = base64.b64encode(grayscale_data).decode()

            return jsonify({"grayscale_image": f"data:image/png;base64,{grayscale_base64}"})
        else:
            return jsonify({"error": "No image uploaded"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
