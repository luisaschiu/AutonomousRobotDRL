import cv2 as cv
import glob
import os

# Test counting number of files in directory
# count = 0
# robot_images = [cv.imread(image) for image in glob.glob("robot_steps/*.jpg")]
# for image in robot_images:
#     # cv.imshow('img', image)
#     # cv.waitKey(1000)
#     count += 1
# print(count)


# Test copying files over
# # Set the current working directory
# os.chdir("C:/Users/luisa/Documents/AutonomousRobotDRL")
# # Set the source and the destination folders
# src = os.getcwd() + "\\robot_steps"
# dst = os.getcwd() + "\\test_images"
# print(src)
# # Copy file
# os.system('copy ' + src+'\\0.jpg ' + dst + '\\0.jpg') # NOTE: can rename file when copying over to dst

# # Deletes a file:
# os.chdir("C:/Users/luisa/Documents/AutonomousRobotDRL")
# dst = os.getcwd() + "\\test_images"
# os.remove(dst +'\\1.jpg')

cur_num = 0
count = 0
os.chdir("C:/Users/luisa/Documents/AutonomousRobotDRL")
src = os.getcwd() + "\\robot_steps"
dst = os.getcwd() + "\\test_images"
while True:
    input_char = input("Click 'y' to generate an image, 'e' to quit: ")
    if input_char == 'y':
        os.system('copy ' + src +'\\' + str(cur_num) + '.jpg ' + dst + '\\' + str(cur_num) + '.jpg')
        images = [cv.imread(image) for image in glob.glob("test_images/*.jpg")]
        for image in images:
            count += 1
        if count > 3:
            print(cur_num)
            os.remove(dst + '\\' + str(cur_num-3) + '.jpg')
        cur_num += 1
        count = 0
    elif input_char == 'e':
        exit()
# TODO: account for edge case for cur_num exceeding number of images in src directory