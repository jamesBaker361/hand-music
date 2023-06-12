import sys
import os
sys.path.append(os.getcwd())
from driver import *
from classifier_src import *
from music_player_src import *
from preprocessor_src import *
import numpy as np
from PIL import Image
#for this application we want to be able to manually look at and inspect the output 
# of some of the code we write

web_cam=cv2.VideoCapture(0)
preprocessor=Preprocessor()
music_player=MusicPlayer()
classifier=Classifier()
driver=Driver(
    1000,
    web_cam,
    classifier,
    music_player,
    preprocessor)

def read_and_save_frame():
    web_cam=cv2.VideoCapture(0)
    time.sleep(1) #it takes a second for the webcam to get ready
    preprocessor=Preprocessor()
    music_player=MusicPlayer()
    classifier=Classifier()
    driver=Driver(
        1,
        web_cam,
        classifier,
        music_player,
        preprocessor)
    raw_img=driver.get_current_frame()
    print(raw_img)
    rgb_frame = cv2.cvtColor(raw_img.astype('uint8'), cv2.COLOR_BGR2RGB)

    # Create a PIL image from the RGB frame
    pil_image = Image.fromarray(rgb_frame)

    # Save the image
    pil_image.save("captured_image.jpg")
    driver.web_cam.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    read_and_save_frame()