import cv2
import numpy as np


def arucode_angle(dictionary, origImg):
    gray = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    markerCorners, markerIds, _ = detector.detectMarkers(gray)

    if markerIds is not None and len(markerIds) > 0:
        for markerRect in markerCorners:
            bottomLeft = markerRect[0][1]
            bottomRight = markerRect[0][0]
            topRight = markerRect[0][3]
            topLeft = markerRect[0][2]
            diff = (topRight[0] - topLeft[0], topLeft[1] - topRight[1])
            angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi
            # normalize values to 0 - 360 degrees
            if angle < 0:
                angle += 360
            # draw text angle of the marker
            angle_deg = round(angle, 2)
    else:
        return None
    return angle_deg

def arucode_location(dictionary, origImg):
    gray = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    markerCorners, markerIds, _ = detector.detectMarkers(gray)

    if markerIds is not None and len(markerIds) > 0:
        for markerRect in markerCorners:
            bottomLeft = markerRect[0][1]
            bottomRight = markerRect[0][0]
            topRight = markerRect[0][3]
            topLeft = markerRect[0][2]
            # draw position in pixels
            center = (bottomRight + topLeft) / 2
    else:
        return None
    return center

def show_arucode_param(dictionary, origImg):
    gray = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    markerCorners, markerIds, _ = detector.detectMarkers(gray)

    if markerIds is not None and len(markerIds) > 0:
        frame = cv2.aruco.drawDetectedMarkers(origImg, markerCorners, markerIds, (0, 0,255))

        for markerRect in markerCorners:
            bottomLeft = markerRect[0][1]
            bottomRight = markerRect[0][0]
            topRight = markerRect[0][3]
            topLeft = markerRect[0][2]
            diff = (topRight[0] - topLeft[0], topLeft[1] - topRight[1])
            angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi
            # normalize values to 0 - 360 degrees
            if angle < 0:
                angle += 360
            width = 5
            # draw text angle of the marker
            angle_deg = round(angle, 2)
            # draw position in pixels
            center = (bottomRight + topLeft) / 2
            cv2.putText(frame,
                        str(angle) + 'deg ',
                        (int(center[0] - 40), int(center[1] + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255),
                        2,
                        cv2.LINE_AA)
            cv2.putText(frame,
                        str(center),
                        (int(center[0] - 40), int(center[1] + 90)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255),
                        2,
                        cv2.LINE_AA)
            length = 50
            # change angle back to radians
            angle = angle_deg * np.pi / 180
            cv2.line(
                frame, 
                (int(center[0]), int(center[1])), 
                (int(center[0] + length * np.cos(angle)), int(center[1] - length * np.sin(angle))),
                (0, 255, 255), 
                5
            )
    cv2.imshow('ArUco Marker Angle', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame

def angle_difference(init_angle, final_angle, tolerance=5):
    # Normalize angles to be within [0, 360)
    init_angle = init_angle % 360
    final_angle = final_angle % 360
    angular_difference = abs(final_angle - init_angle)

    # Special case for handling 270 to 0 and 0 to 270
    if angular_difference < tolerance:
        direction = "No Rotation"
    elif (260 <= init_angle <= 280) and (0 <= final_angle < 10):
        angular_difference = abs(final_angle-init_angle)-180
        direction = "CCW"
    elif (260 <= final_angle <= 280) and (0 <= init_angle < 10):
        angular_difference = abs(final_angle-init_angle)-180
        direction = "CW"
    elif angular_difference > 180:
        angular_difference = angular_difference-180
        if init_angle > final_angle:
            direction = "CCW"
        elif init_angle < final_angle:
            direction = "CW"
    else:
        counterclockwise_path = (final_angle - init_angle + 360) % 360
        clockwise_path = (init_angle - final_angle + 360) % 360
        direction = "CW" if clockwise_path < counterclockwise_path else "CCW"
    return angular_difference, direction


def find_grid_position(img, x, y):
    w, h = (img.shape[1], img.shape[0])
    if (x < int(w/4+14)) and (y < int(h/4-5)):
        return (0, 0)
    elif (int(w/4+14) < x < int(w*2/4)) and (y < int(h/4-5)):
        return (0, 1)
    elif (int(w*2/4) < x < int(w*3/4)) and (y < int(h/4-5)):
        return (0, 2)
    elif (int(w*3/4) < x < int(w)) and (y < int(h/4-5)):
        return (0, 3)
    elif (x < int(w/4+14)) and (int(w/4-5) < y < int(h*2/4-10)):
        return (1, 0)
    elif (int(w/4+14) < x < int(w*2/4)) and (int(h/4-5) < y < int(h*2/4-10)):
        return (1, 1)
    elif (int(w*2/4) < x < int(w*3/4)) and (int(h/4-5) < y < int(h*2/4-10)):
        return (1, 2)
    elif (int(w*3/4) < x < int(w)) and (int(h/4-5) < y < int(h*2/4-10)):
        return (1, 3)
    elif (x < int(w/4+14)) and (int(h*2/4-10) < y < int(h*3/4-10)):
        return(2, 0)
    elif (int(w/4+14) < x < int(w*2/4)) and (int(h*2/4-10) < y < int(h*3/4-10)):
        return(2, 1)
    elif (int(w*2/4) < x < int(w*3/4)) and (int(h*2/4-10) < y < int(h*3/4-10)):
        return(2, 2)
    elif (int(w*3/4) < x < int(w)) and (int(h*2/4-10) < y < int(h*3/4-10)):
        return(2, 3) 
    elif (x < int(w/4+14)) and (int(h*3/4-10) < y < int(h)):
        return(3, 0)
    elif (int(w/4+14) < x < int(w*2/4)) and (int(h*3/4-10) < y < int(h)):
        return(3, 1)
    elif (int(w*2/4) < x < int(w*3/4)) and (int(h*3/4-10) < y < int(h)):
        return(3, 2)
    elif (int(w*3/4) < x < int(w)) and (int(h*3/4-10) < y < int(w)):
        return(3, 3) 
    else:
        return None
    
def draw_gridlines (origImg):
    # origImg = cv2.imread(inputImg)
    w, h = origImg.shape[1], origImg.shape[0]
    # Draw Vertical Lines
    cv2.line(
        origImg, 
        (int(w/4+14), int(0)), 
        (int(w/4+14), int(h)),
        (0, 0, 255), 
        1
    )
    cv2.line(
        origImg, 
        (int(w*2/4), int(0)), 
        (int(w*2/4), int(h)),
        (0, 0, 255),  
        1
    )
    cv2.line(
        origImg, 
        (int(w*3/4), int(0)), 
        (int(w*3/4), int(h)),
        (0, 0, 255),  
        1
    )
    # Draw Horizontal Lines
    cv2.line(
        origImg, 
        (int(0), int(w/4-5)), 
        (int(h), int(w/4-5)),
        (0, 0, 255), 
        1
    )
    cv2.line(
        origImg, 
        (int(0), int(w*2/4-10)), 
        (int(h), int(w*2/4-10)),
        (0, 0, 255), 
        1
    )
    cv2.line(
        origImg, 
        (int(0), int(w*3/4-10)), 
        (int(h), int(w*3/4-10)),
        (0, 0, 255), 
        1
    )

    cv2.imshow('gridlines', origImg)
    # cv2.imwrite('gridlines.jpg', origImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()