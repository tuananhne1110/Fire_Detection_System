import cv2
import numpy as np

# Define the algorithm as described
def enhanced_fire_detection(image_path):
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
    final_image = cv2.cvtColor(combined_image, cv2.COLOR_HSV2BGR)
    
    return final_image

# Assuming the image path from the uploaded file
image_path = '/test.jpg'

# Run the algorithm
final_image = enhanced_fire_detection(image_path)

# Save the final image
output_path = '/enhanced_image.png'
cv2.imwrite(output_path, final_image)

output_path

def enhance_hsv(image, lower_bound1=(0, 43, 46), upper_bound1=(10, 255, 255),
                         lower_bound2=(170, 43, 46), upper_bound2=(179, 255, 255),
                         weight_original=0.7, weight_heatmap=0.3):
    # Convert RGB image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create binary masks for both sets of boundaries
    mask1 = cv2.inRange(hsv_image, np.array(lower_bound1), np.array(upper_bound1))
    mask2 = cv2.inRange(hsv_image, np.array(lower_bound2), np.array(upper_bound2))

    # Combine masks using bitwise OR operation
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Convert binary mask to heatmap
    heatmap = cv2.applyColorMap(combined_mask, cv2.COLORMAP_HOT)

    # Convert heatmap to BGR for weighted addition
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # Weighted addition of original image and heatmap
    enhanced_image = cv2.addWeighted(image, weight_original, heatmap_bgr, weight_heatmap, 0)

    return enhanced_image
