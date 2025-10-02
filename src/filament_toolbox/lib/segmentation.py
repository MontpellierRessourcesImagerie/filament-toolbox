import numpy as np
from skimage.segmentation import clear_border


class Segmentation(object):


    def __init__(self, image):
        super().__init__()
        self.image = image
        self.result = None
        
        
        
class Threshold(Segmentation):
    
    
    def __init__(self, image):
        super().__init__(image)
        self.min_value = 128
        self.max_value = None


    def run(self):
        self.result = np.zeros(self.image.shape, dtype=np.uint8)
        if not self.min_value is None:
            indices = np.where(self.image >= self.min_value)
            self.result[indices] = 255
        if not self.max_value is None:
            indices = np.where(self.image > self.max_value)
            self.result[indices] = 0



class ClearBorder(Segmentation):


    def __init__(self, image):
        super().__init__(image)


    def run(self):
        self.result = clear_border(self.image)


