import cv2
import numpy as np
import cvzone
import math 
from ultralytics import YOLO


model = YOLO('D:\\Fire_Detection_System\\yolo\\train\\Weights\\best.pt')

# Reading the classes
classnames = ['Fire', 'Smoke']

def predict_and_enhance(image_input):
    # Enhance the image using the HSV function
    enhanced_image = enhanced_fire_detection(image_input)

    # Get results from the YOLO model on the enhanced image
    results = model(enhanced_image, stream=True)

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
                cv2.rectangle(enhanced_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(enhanced_image, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    return enhanced_image

# Example usage
image_path = "test.jpg"
original_image = cv2.imread(image_path)

# Predict and enhance the image
result_image = predict_and_enhance(original_image)

# Display the result image
cv2.imshow("Result Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
