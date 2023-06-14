import torch
import cv2 as cv

from functools import reduce
import os, sys

# utility function
def compose(*fns):
    def __inner__(*args):
        f = lambda acc, fn: fn(acc)
        return reduce(f, fns, *args)
    return __inner__

""" all img read functions """
blur = lambda img: cv.GaussianBlur(img, (7, 7), 0) 
canny = lambda img: cv.Canny(img, 30, 100)
bnw = lambda img: cv.cvtColor(img, cv.COLOR_BGR2GRAY)

class Preprocessor:
    def __init__(self, img_path: str):
        img = os.path.expanduser(img_path)
        assert os.path.exists(img_path), "img file path doesn't exist"
        self.img = cv.imread(img_path)

    def capture(self, *fns):
        vid_process = compose(*fns)
        def __inner__(file: str|None=None):
            vid = cv.VideoCapture(file if file is not None else 0)
            # init prev and curr frame
            _, prev = vid.read()
            _, curr = vid.read()

            while True:
                prev = curr
                ret, curr = vid.read()
                if not ret or cv.waitKey(60) == 27:
                    break

                if len(fns) == 0:
                    cv.imshow("sample", cv.absdiff(prev, curr))
                else:
                    cv.imshow("sample", vid_process(curr))
            # cleanup
            vid.release()
            cv.destroyAllWindows()
        return __inner__

    def to_tensor(self):
        return torch.Tensor(self.img)
