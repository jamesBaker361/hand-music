import torch

class Preprocessor:
    def __init__(self):
        pass

    def process_image(self,raw_img):
        #THIS METHOD IS A STUB; TODO: IMPLEMENT IT
        return torch.randn(1,3,128,128) #random white noise image in pytorch shape = (B,C,H,W)