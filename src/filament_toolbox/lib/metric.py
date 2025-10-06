from abc import abstractmethod
import numpy as np
from skimage.morphology import skeletonize


class Metric(object):


    def __init__(self, labels1, labels2):
        super().__init__()
        self.labels1 = labels1
        self.labels2 = labels2
        self.mask1 = None
        self.mask2 = None
        self.result = -1


    @abstractmethod
    def calculate(self):
        raise(Exception("Abstract method calculate of class Metric called!"))


    def calculate_masks(self):
        new_shape = np.maximum(np.array(self.labels1.shape), np.array(self.labels2.shape))
        self.mask1 = np.zeros(new_shape, np.uint8)
        self.mask2 = np.zeros(new_shape, np.uint8)
        self.mask1[np.where(self.labels1 > 0)] = 1
        self.mask2[np.where(self.labels2 > 0)] = 1



class Dice(Metric):


    def __init__(self, labels1, labels2):
        super().__init__(labels1, labels2)


    def calculate(self):
        self.calculate_masks()
        intersection = self.mask1 * self.mask2
        cardinality_intersection = len(np.where(intersection > 0)[0])
        cardinality_mask1 = len(np.where(self.mask1 > 0)[0])
        cardinality_mask2 = len(np.where(self.mask2 > 0)[0])
        self.result = (2 * cardinality_intersection) / (cardinality_mask1 + cardinality_mask2)



class CenterlineDice(Metric):


    def __init__(self, labels1, labels2):
        super().__init__(labels1, labels2)


    def calculate(self):
        self.calculate_masks()
        topological_precision = self.cl_score(self.mask1, skeletonize(self.mask2))
        topological_sensitivity = self.cl_score(self.mask2, skeletonize(self.mask1))
        self.result = 2 * topological_precision * topological_sensitivity / (topological_precision + topological_sensitivity)


    @classmethod
    def cl_score(cls, volume, skeleton):
        return np.sum(volume * skeleton) / np.sum(skeleton)

