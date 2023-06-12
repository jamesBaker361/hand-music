import sys
import os
import torch
sys.path.append(os.getcwd())
from driver import *
from classifier_src import *
from music_player_src import *
from preprocessor_src import *
import random
import numpy as np
import unittest

from unittest.mock import patch,MagicMock

web_cam=cv2.VideoCapture(0)
preprocessor=Preprocessor()
music_player=MusicPlayer()
classifier=Classifier()


class DriverTestCase(unittest.TestCase):
    
    def setUp(self):
        self.driver=Driver(
            0.01,
            web_cam,
            classifier,
            music_player,
            preprocessor
        )

    
    def test_get_current_frame(self):
        frame=self.driver.get_current_frame()
        assert isinstance(frame, np.ndarray), "Variable is not of type numpy.ndarray"

    @patch.object(preprocessor, "process_image",return_value=torch.randn(1,3,128,128))
    @patch.object(music_player, 'decide_action', return_value=None)
    @patch.object(classifier, "classify_image", return_value=("10101", 0.95))
    def test_main(self,classifier_mock, music_player_mock,preprocessor_mock):
        max_frames=4
        self.driver.max_frames=max_frames
        self.driver.main()
        assert preprocessor_mock.call_count==max_frames, 'process_image not called {} times'.format(max_frames)
        assert music_player_mock.call_count==max_frames, 'decide_action not called {} times'.format(max_frames)
        assert classifier_mock.call_count==max_frames, 'classify_image not called {} times'.format(max_frames)


if __name__=='__main__':
    unittest.main() # run all tests