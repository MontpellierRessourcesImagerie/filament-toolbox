from skimage.morphology import binary_dilation, remove_small_objects
from filament_toolbox.lib.filter import FilterWithSE, Filter
from skimage.morphology import binary_closing
from skimage.measure import label


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