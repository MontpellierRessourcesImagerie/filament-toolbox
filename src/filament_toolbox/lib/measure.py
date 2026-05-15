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
        self.selectedProperties = {"label", "area"}

    @classmethod
    def getAllProperties(cls):
        return (
            cls.getProperties()
            + cls.getIntensityProperties()
            + cls.get2DOnlyProperties()
            + cls.getIntensity2DOnlyProperties()
        )

    @classmethod
    def get2DOnlyProperties(cls):
        return (
            "eccentricity",
            "moments_hu",
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
            "equivalent_diameter_area",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "moments",
            "moments_central",
            "moments_normalized",
            "solidity",
        )

    @classmethod
    def getIntensityProperties(cls):
        return (
            "centroid_weighted",
            "centroid_weighted_local",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "intensity_std",
            "moments_weighted",
            "moments_weighted_central",
            "moments_weighted_normalized",
        )

    @classmethod
    def getIntensity2DOnlyProperties(cls):
        return ("moments_weighted_hu",)

    def run(self):
        print("intensity image", self.intensityImage)
        properties = self.selectedProperties.copy()
        if self.intensityImage is None:
            properties = properties - set(self.getIntensityProperties())
            properties = properties - set(self.getIntensity2DOnlyProperties())
        if self.labels.ndim > 2:
            properties = properties - set(self.get2DOnlyProperties())
            properties = properties - set(self.getIntensity2DOnlyProperties())
        self.table = regionprops_table(
            self.labels,
            properties=properties,
            intensity_image=self.intensityImage,
            spacing=self.scale,
        )
