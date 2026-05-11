from skimage.transform import rescale


class IsotropicResampling(object):

    def __init__(self, inputImage, voxelSize):
        super().__init__()
        self.image = inputImage
        self.voxelSize = voxelSize
        self.result = None

    def run(self):
        zScaleFactor = self.voxelSize[0] / self.voxelSize[1]
        scaling = (zScaleFactor, 1, 1)
        self.result = rescale(self.image, scaling)
