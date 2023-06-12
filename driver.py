import cv2
import time
from classifier_src import *
from music_player_src import *
from preprocessor_src import *

INTERVAL = 1000 #read from webcam every INTERVAL # of milliseconds

class Driver:
    def __init__(self,interval,
                 web_cam,
                 classifier,
                 music_player,
                 preprocessor,
                 max_frames=-1
                 ):
        self.interval=interval
        self.web_cam=web_cam
        self.classifier=classifier
        self.music_player=music_player
        self.preprocessor=preprocessor
        self.max_frames=max_frames


    def get_current_frame(self):
        success, raw_img = self.web_cam.read()
        if success:
            return raw_img
        else:
            raise Exception("could not successfully get current frame from web_cam")
    
    def main(self):
        frames=0
        while True:
            raw_img=self.get_current_frame()
            processed_image=self.preprocessor.process_image(raw_img)
            class_label, confidence_score= self.classifier.classify_image(processed_image)
            self.music_player.decide_action(class_label, confidence_score)
            time.sleep(self.interval)
            frames+=1
            if self.max_frames!=-1 and frames >=self.max_frames:
                break
        self.web_cam.release()
        cv2.destroyAllWindows()

