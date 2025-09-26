from abc import abstractmethod
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from filament_toolbox.lib.ext.fastaniso import anisodiff, anisodiff3


class Filter(object):


    def __init__(self, input_image):
        self.image = input_image
        self.mode='reflect'
        self.result = None


    @abstractmethod
    def run(self):
        raise Exception("Abstract method of class Filter called!")



class FilterWithSE(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.size = (3, 3, 3)
        self.footprint = None


    @abstractmethod
    def run(self):
        raise Exception("Abstract method of class FilterWithSE called!")


    def get_size(self):
        if self.image.ndim == 2:
            return self.size[1:]
        return self.size



class MedianFilter(FilterWithSE):


    def __init__(self, input_image):
        super().__init__(input_image)


    def run(self):
        self.result = median_filter(self.image,
                                    size=self.get_size(),
                                    footprint=self.footprint,
                                    mode=self.mode
                                    )
        


class GaussianFilter(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.sigma = (1.3, 1.3, 1.3)


    def run(self):
        self.result = gaussian_filter(self.image, self.sigma, mode=self.mode)



class AnisotropicDiffusionFilter(Filter):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.niter = 5
        self.kappa = 50
        self.gamma = 0.1
        self.step = (1.,1.,1.)
        self.option = 1


    def get_step(self):
        if self.image.ndim == 2:
            return self.step[1:]
        return self.step


    def run(self):
        if self.image.ndim == 2:
            self.result = anisodiff(self.image,
                      niter=self.niter,
                      kappa=self.kappa,
                      gamma=self.gamma,
                      step=self.get_step(),
                      option=self.option)
        else:
            self.result = anisodiff3(self.image,
                      niter=self.niter,
                      kappa=self.kappa,
                      gamma=self.gamma,
                      step=self.get_step(),
                      option=self.option)

