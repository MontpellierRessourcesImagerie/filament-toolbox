from skan import Skeleton
from skan import summarize
from skimage.measure import regionprops_table


class MeasureSkeleton(object):

    def __init__(self, mask):
        super().__init__()
        self.image = mask
        self.result = None
        self.result_image = None
        self.scale = [1] * mask.ndim
        self.units = ["pixel"] * mask.ndim

    def run(self):
        skeleton = Skeleton(self.image, spacing=self.scale)
        branch_data = summarize(skeleton, separator="_", find_main_branch=True)
        self.result = {}
        for key, value in branch_data.items():
            self.result[key] = value.values
        self.result_image = skeleton.path_label_image()


class MeasureLabels(object):

    def __init__(self, labels, intensityImage=None, scale=(1, 1, 1)):
        super().__init__()
        self.labels = labels
        self.intensityImage = intensityImage
        self.scale = scale
        self.table = None

    @classmethod
    def getAllProperties(cls):
        return cls.getProperties() + cls.get2DOnlyProperties()

    @classmethod
    def get2DOnlyProperties(cls):
        return (
            "eccentricity",
            "moments_hu",
            "moments_weighted_hu",
            "orientation",
            "perimeter",
            "perimeter_crofton",
        )

    @classmethod
    def getProperties(cls):
        return (
            "label",
            "area",
            "area_bbox",
            "area_convex",
            "area_filled",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
            "centroid",
            "centroid_local",
            "centroid_weighted",
            "centroid_weighted_local",
            "equivalent_diameter_area",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "intensity_std",
            "label",
            "moments",
            "moments_central",
            "moments_normalized",
            "moments_weighted",
            "moments_weighted_central",
            "moments_weighted_normalized",
            "solidity",
        )

    def run(self):
        properties = self.getProperties()
        if self.labels.ndim == 2:
            properties = self.getAllProperties()
        self.table = regionprops_table(
            self.labels,
            properties=properties,
            intensity_image=self.intensityImage,
            spacing=self.scale,
        )
