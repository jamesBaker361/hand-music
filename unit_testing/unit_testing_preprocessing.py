import sys, os, torch, cv2 as cv
sys.path.append(os.getcwd())

import unittest
from preprocessor_src import *

class PreprocessorTestCase(unittest.TestCase):
    def setUp(self):
        path = "~/codebase/dl-research/hand-music/unit_testing/img_samples"
        self.imgs = ["/gesture-I.jpeg", "/gesture-II.jpeg", "/gesture-III.jpeg" ]
        self.imgs = list(map(lambda x: cv.imread(os.path.expanduser(path + x)), self.imgs))
        self.p = Preprocessor()

    def test_returntensor(self):
        for img in self.imgs:
            self.assertEqual(type(self.p.process_image(img)), torch.Tensor)

    def maintains_dimensions(self):
        for img in self.imgs:
            self.assertEqual(self.p.process_image(img).shape[:2], (448, 448))

    def handles_typechanges(self):
        tensorInstance = lambda x: isinstance(x, torch.Tensor)
        for img in self.imgs:
            self.assertEqual(self.p.process_image(img).shape[:2], (448, 448))
            self.assertTrue(tensorInstance(self.p.process_image(img)))
            self.assertTrue(tensorInstance(self.p.augment(rotate, flip, grayscale)(img)))

if __name__ == "__main__":
    unittest.main()
