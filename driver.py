import cv2

INTERVAL = 1000 #read from webcam every INTERVAL # of milliseconds

def get_current_frame(web_cam):
    success, frame = web_cam.read()
    if success:
        return frame
    else:
        raise Exception("could not successfully get current frame from web_cam")
    