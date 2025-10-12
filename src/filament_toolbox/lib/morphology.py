import numpy as np
import cv2
from skimage.morphology import binary_dilation, remove_small_objects
from filament_toolbox.lib.filter import FilterWithSE, Filter
from skimage.morphology import binary_closing, skeletonize
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import localthickness as lt
try:
    from pyhjs import PyHJS, BinaryFrame
except Exception as e:
    print(f"Could not import PyHJS: {e}")



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
        # contour_mask = self.get_binary_image_contour(self.image)
        # skeleton = self.pruning_skeleton_mask(skeleton, contour_mask, edge_redundant_threshold=30)
        self.result = skeleton


    def get_separate_skeleton_mask(self, skeleton_image):
        # TODO: rename img_kel
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        img_kel = cv2.filter2D(skeleton_image, -1, kernel)
        img_kel[skeleton_image == 0] = 0
        img_jc = np.zeros_like(img_kel)
        img_jc[img_kel > 2] = 1
        img_bridge = np.zeros_like(img_kel)
        img_bridge[(img_kel == 1) | (img_kel == 2)] = 1
        img_end_point = np.zeros_like(img_kel)
        img_end_point[img_kel == 1] = 1
        return img_jc, img_bridge, img_end_point


    def pruning_skeleton_mask(self, skeleton_image, contour_mask, dilate_kernel_size=9,
                              edge_redundant_threshold=50):
        # Get below masks
        # - end points
        # - junction points
        # - bridge area
        _, img_bridge, _ = self.get_separate_skeleton_mask(skeleton_image)

        # Calculate connnected component algorithm against bridge_image
        _, cc_bridge_labels = cv2.connectedComponentsWithAlgorithm(img_bridge.astype(np.uint8), connectivity=8,
                                                                   ltype=cv2.CV_16U, ccltype=cv2.CCL_SAUF)

        # Removal redundant labels
        contour_mask_dilated = cv2.dilate(contour_mask, np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8))
        redundant_labels = np.unique(cc_bridge_labels[contour_mask_dilated > 0])
        label_mask_filtered = skeleton_image.copy()
        for redundant_label in redundant_labels:
            if redundant_label == 0:
                continue
            if np.sum(cc_bridge_labels == redundant_label) < edge_redundant_threshold:
                label_mask_filtered[cc_bridge_labels == redundant_label] = 0
        return label_mask_filtered


    ### (2) Redundant skeleton-edge removal
    # build graph from skeleton
    def get_binary_image_contour(self, binary_image):
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_mask = np.zeros_like(binary_image)
        for contour in contours:
            contour_mask[contour[:, 0, 1], contour[:, 0, 0]] = 255
        return contour_mask



class EuclideanDistanceTransform(Filter):


    def __init__(self, image):
        super().__init__(image)


    def run(self):
        self.result = distance_transform_edt(self.image)



class LocalThickness(Filter):


    def __init__(self, image):
        super().__init__(image > 0)
        self.scale = 0.5
        self.histogram = None


    def run(self):
        self.result = lt.local_thickness(self.image, scale=self.scale)



