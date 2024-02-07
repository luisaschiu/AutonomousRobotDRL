import cv2

def preprocess_image(image):
    # Convert RGB image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Extract Y channel (luminance)
    y_channel = yuv_image[:,:,0]
    
    # Resize Y channel to 84x84
    resized_image = cv2.resize(y_channel, (150, 150))
    
    return resized_image

if __name__ == "__main__":
    image = cv2.imread('robot_steps/0.jpg')
    processed_img = preprocess_image(image)
    cv2.imshow('img', processed_img)
    cv2.waitKey(0)