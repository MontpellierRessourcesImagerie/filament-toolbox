import numpy as np
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial



class RandomForestPixelClassifier(object):


    def __init__(self, image):
        super().__init__()
        self.image = image
        self.intensity = False
        self.edges = True
        self.texture = True
        self.sigma_min = 1
        self.sigma_max = 16
        self.channel_axis = None     # None for single channel images
        self.training_labels = np.zeros(self.image.shape, dtype=np.uint8)   # zeros mean as yet unclassified
        self.features = None
        self.classifier = None
        self.features_func = None
        self.result = None
        self.n_estimators = 50
        self.n_jobs = -1
        self.max_depth = 10
        self.num_sigma = None
        self.training_points = None
        self.training_points_classes = None


    def train(self):
        self.calculate_training_labels()
        self.features_func = partial(
            feature.multiscale_basic_features,
            intensity=self.intensity,
            edges=self.edges,
            texture=self.texture,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_sigma=self.num_sigma,
            channel_axis=self.channel_axis,
        )
        self.features = self.features_func(self.image)
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators,
                                                 n_jobs=self.n_jobs,
                                                 max_depth=self.max_depth)
        self.classifier = future.fit_segmenter(self.training_labels, self.features, self.classifier)


    def predict(self):
        features_new = self.features_func(self.image)
        self.result = future.predict_segmenter(features_new, self.classifier)


    def calculate_training_labels(self):
        points = self.training_points
        self.training_labels = np.zeros(self.image.shape, np.uint8)
        classes = list(set(self.training_points_classes))
        if self.image.ndim == 2:
            for index, (y, x) in enumerate(points):
                self.training_labels[int(round(y))][int(round(x))] = classes.index(self.training_points_classes[index]) + 1
        if self.image.ndim == 3:
            for index, (z, y, x) in enumerate(points):
                self.training_labels[int(round(z))][int(round(y))][int(round(x))] = classes.index(self.training_points_classes[index]) + 1

