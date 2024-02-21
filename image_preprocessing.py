import cv2
import tensorflow as tf

def preprocess_image(image):
    # Convert RGB image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Extract Y channel (luminance)
    y_channel = yuv_image[:,:,0]
    
    # Resize Y channel to 84x84
    # resized_image = cv2.resize(y_channel, (150, 150))
    resized_image = cv2.resize(y_channel, (84, 84))
    return resized_image

if __name__ == "__main__":
    image = cv2.imread('robot_steps/0.jpg')
    print(image.shape)
    processed_img = preprocess_image(image)
    cv2.imshow('img', processed_img)
    cv2.waitKey(0)
    print(processed_img.shape)
    # print(tf.constant((processed_img), dtype = tf.float32))