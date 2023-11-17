import cv2
import numpy as np
import cvzone
import math 
from ultralytics import YOLO


def enhanced_fire_detection(image):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert Image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the HSV boundaries for red color (flame color)
    lower_bound1 = np.array([0, 43, 46])
    upper_bound1 = np.array([10, 255, 255])
    lower_bound2 = np.array([170, 43, 46])
    upper_bound2 = np.array([179, 255, 255])
    
    # Generate masks based on the HSV boundaries
    mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)
    mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)
    
    # Combine the masks to capture all the red features in the image
    combined_mask = cv2.bitwise_or(mask1, mask2)
    
    # Generate the enhanced HSV image by combining the original with the mask
    enhanced_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=combined_mask)
    
    # Overlay the mask on the original image
    combined_image = cv2.addWeighted(image, 0.7, enhanced_hsv, 0.3, 0)
    
    # Convert the final image back from HSV to RGB color space before saving/displaying
    enhanced_image = cv2.cvtColor(combined_image, cv2.COLOR_HSV2BGR)
    
    return enhanced_image
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
