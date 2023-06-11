import sys
import os
sys.path.append(os.getcwd())
from driver import *
import numpy as np
import unittest

class SimpleTestCase(unittest.TestCase):
    def setUp(self):
        self.web_cam=cv2.VideoCapture(0)

    def test_get_current_frame(self):
        frame=get_current_frame(self.web_cam)
        assert isinstance(frame, np.ndarray), "Variable is not of type numpy.ndarray"

if __name__=='__main__':
    unittest.main() # run all tests