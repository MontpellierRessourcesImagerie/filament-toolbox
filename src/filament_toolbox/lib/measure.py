from skan import Skeleton, summarize



class MeasureSkeleton(object):


    def __init__(self, mask):
        super().__init__()
        self.image = mask
        self.result = None
        self.result_image = None
        self.scale = [1] * mask.ndim
        self.units = ['pixel'] * mask.ndim


    def run(self):
        skeleton = Skeleton(self.image, spacing=self.scale)
        branch_data = summarize(skeleton, separator='_')
        self.result = {}
        for key, value in branch_data.items():
            self.result[key] = value.values
        self.result_image = skeleton.path_label_image()