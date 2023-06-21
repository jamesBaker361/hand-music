import torch
from functools import reduce
import cv2 as cv

# blur = lambda img: cv.GaussianBlur(img, (7, 7), 0) 
# canny = lambda img: cv.Canny(img, 30, 100)
def compose(*fns):
    return lambda x: reduce(lambda acc, fn: fn(acc), fns, x)

def grayscale(img):
    """ converts RGB img to grayscale """
    if len(img.shape) == 2:
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def rotate(img, angle):
    """ rotates <arg1> by <arg2> degrees """
    (h, w) = img.shape[:2]
    center = (w >> 1, h >> 1)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1)
    return cv.warpAffine(img, rot_mat, (w, h))

def flip(img):
    """ flips <arg> on its side """
    return cv.flip(img, 1)

def to_numpy(img):
    return img.numpy() if type(img) == torch.Tensor else img

class Preprocessor:
    @staticmethod
    def resize(img):
        """
        resizes image to 448x448 i.e, pixel ratio in original YOLO model
        """
        cv.resize(img, (448, 448), interpolation=cv.INTER_AREA)

    @staticmethod
    def augment(*fns):
        return compose(to_numpy, *fns, grayscale, torch.Tensor)

    def process_image(self, img):
        if img.shape[0] != 448 and img.shape[1] != 448:
            Preprocessor().resize(img)
        return torch.Tensor(img)

"""
example USAGE

img = cv.imread('/Users/prateekpravanjan/Downloads/collection/khabibxconor.jpeg')

p = Preprocessor()

rotate45 = lambda img: rotate(img, 45)

img = p.process_image(img)
img = p.augment(rotate45)(img)

print(img)
"""