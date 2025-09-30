from skimage.morphology import binary_dilation
from filament_toolbox.lib.filter import FilterWithSE



class Dilation(FilterWithSE):


    def __init__(self, input_image):
        super().__init__(input_image)


    def run(self):
        self.result = binary_dilation(self.image,
                                    footprint=self.footprint,
                                    mode=self.mode
                                    )