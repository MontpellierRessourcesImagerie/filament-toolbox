import numpy as np
from skimage.morphology import binary_dilation, remove_small_objects
from filament_toolbox.lib.filter import FilterWithSE, Filter
from skimage.morphology import binary_closing, skeletonize
from skimage.measure import label
from pyhjs import PyHJS, BinaryFrame


class Dilation(FilterWithSE):


    def __init__(self, input_image):
        super().__init__(input_image)


    def run(self):
        self.result = binary_dilation(self.image,
                                    footprint=self.footprint,
                                    mode=self.mode
                                    )



class Closing(FilterWithSE):


    def __init__(self, input_image):
        super().__init__(input_image)


    def run(self):
        self.result = binary_closing(self.image,
                                    footprint=self.footprint,
                                    mode=self.mode
                                    )



class Label(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.connectivity = input_image.ndim


    def run(self):
        self.result = label(self.image, connectivity=self.connectivity)



class RemoveSmallObjects(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.min_size = 64


    def run(self):
        self.result = remove_small_objects(self.image, min_size=self.min_size)



class Skeletonize(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.method = "zhang"
        self.methods = ["lee", "zhang"]


    def run(self):
        self.result = skeletonize(self.image, method=self.method)



class HamiltonJacobiSkeleton(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.flux_threshold = 2.5       # gamma
        self.dilation = 1.5             # epsilon
        self.use_anisotropic_diffusion = False


    def run(self):
        print("HJS", "threshold", self.flux_threshold, "dilation", self.dilation)
        hjs = PyHJS(self.flux_threshold, self.dilation)
        frame = BinaryFrame(self.image)
        hjs.compute(frame, enable_anisotropic_diffusion=self.use_anisotropic_diffusion)
        skeleton_raw =  hjs.get_skeleton_image()
        skeleton = np.zeros(skeleton_raw.shape, np.uint8)
        skeleton[skeleton_raw > 0] = 1
        self.result = skeleton