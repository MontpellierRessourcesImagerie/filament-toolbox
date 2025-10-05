"""
This module contains the tools of the filament-toolbox
"""
from codecs import namereplace_errors
from typing import TYPE_CHECKING

import numpy as np
import napari
from napari.layers import Image, Labels
from numba.core.types import uint16, uint32
from qtpy.QtWidgets import QPushButton, QWidget
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout
from qtpy.QtWidgets import QSlider, QCheckBox
from qtpy.QtCore import Qt
import skimage.morphology
from skimage.color import rgb2gray
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import notifications
from filament_toolbox.lib.qtutil import WidgetTool
from filament_toolbox.lib.napari_util import NapariUtil
from filament_toolbox.lib.filter import MedianFilter, GaussianFilter, AnisotropicDiffusionFilter, RollingBall
from filament_toolbox.lib.filter import FrangiFilter, SatoFilter, MeijeringFilter
from filament_toolbox.lib.segmentation import Threshold, ClearBorder
from filament_toolbox.lib.morphology import Dilation, Closing, Label, RemoveSmallObjects, Skeletonize
from filament_toolbox.lib.morphology import HamiltonJacobiSkeleton
from filament_toolbox.lib.ml import RandomForestPixelClassifier

if TYPE_CHECKING:
    import napari


def activate():
    print("Filament Toolbox activated")


def rgb_to_8bit( viewer: "napari.viewer.Viewer"):
    layer = viewer.layers.selection.active
    name = layer.name + " 8bit"
    converted = rgb2gray(layer.data)
    converted = (converted * 255).astype(np.uint8)
    viewer.add_image(
        converted,
        name=name,
        scale=layer.scale,
        units=layer.units,
        blending='additive',
        colormap='gray'
    )



def rgb_to_16bit( viewer: "napari.viewer.Viewer"):
    layer = viewer.layers.selection.active
    name = layer.name + " 16bit"
    converted = rgb2gray(layer.data)
    converted = (converted * 65535).astype(np.uint16)
    viewer.add_image(
        converted,
        name=name,
        scale=layer.scale,
        units=layer.units,
        blending='additive',
        colormap='gray'
    )



def str_to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None



class ToolboxWidget(QWidget):


    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.field_width = 50
        self.napari_util = NapariUtil(self.viewer)
        self.image_layers = self.napari_util.getImageLayers()
        self.label_layers = self.napari_util.getLabelLayers()
        self.image_combo_boxes = []
        self.label_combo_boxes = []
        self.point_combo_boxes = []
        self.input_layer_combo_box = None
        self.label_layer_combo_box = None
        self.footprints = ["none", "cube", "ball", "octahedron"]
        self.footprint = "cube"
        self.modes = ["reflect", "constant", "nearest", "mirror", "warp"]
        self.mode = "reflect"
        self.footprint_combo_box = None
        self.footprint_radius_input = None
        self.mode_combo_box = None
        self.input_layer = None
        self.filter = None
        self.viewer.layers.events.inserted.connect(self.on_layer_added_or_removed)
        self.viewer.layers.events.removed.connect(self.on_layer_added_or_removed)


    def on_layer_added_or_removed(self, event: Event):
        self.update_layer_selection_combo_boxes()


    def update_layer_selection_combo_boxes(self):
        image_layers = self.napari_util.getImageLayers()
        label_layers = self.napari_util.getLabelLayers()
        point_layers = self.napari_util.getPointsLayers()
        for combo_box in self.image_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, image_layers)
        for combo_box in self.label_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, label_layers)
        for combo_box in self.point_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, point_layers)


    @classmethod
    def get_footprint(cls, name, radius, dims):
        se_name = name
        two_d_ses = {'ball': 'disk', 'octahedron': 'diamond'}
        if dims == 2:
            if name in two_d_ses.keys():
                se_name = two_d_ses[name]
        if name == "cube":
            footprint_width = 2 * radius + 1
            if dims == 2:
                footprint = skimage.morphology.footprint_rectangle((footprint_width, footprint_width))
            else:
                footprint = skimage.morphology.footprint_rectangle((footprint_width, footprint_width, footprint_width))
        else:
            footprint_function = getattr(skimage.morphology, se_name)
            footprint = footprint_function(radius)
        return footprint



# noinspection PyTypeChecker
class MedianFilterWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.setWindowTitle("Median Filter")
        self.viewer = viewer
        self.median_size_xy = 3
        self.median_size_z = 1
        self.footprint_radius = 1
        self.median_size_xy_input = None
        self.median_size_z_input = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        median_size_xy_label, self.median_size_xy_input = WidgetTool.getLineInput(self, "size xy:",
                                                                   self.median_size_xy,
                                                                   self.field_width,
                                                                   self.median_size_changed)
        median_size_z_label, self.median_size_z_input = WidgetTool.getLineInput(self, "size z:",
                                                                   self.median_size_z,
                                                                   self.field_width,
                                                                   self.median_size_changed)
        footprint_label, self.footprint_combo_box = WidgetTool.getComboInput(self, "footprint:",
                                                                                 self.footprints,
                                                                                 )
        footprint_radius_label, self.footprint_radius_input = WidgetTool.getLineInput(self, "radius:",
                                                                   self.footprint_radius,
                                                                   self.field_width,
                                                                   self.footprint_radius_changed)
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                                 self.modes,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        size_layout = QHBoxLayout()
        footprint_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        size_layout.addWidget(median_size_xy_label)
        size_layout.addWidget(self.median_size_xy_input)
        size_layout.addWidget(median_size_z_label)
        size_layout.addWidget(self.median_size_z_input)
        footprint_layout.addWidget(footprint_label)
        footprint_layout.addWidget(self.footprint_combo_box)
        footprint_layout.addWidget(footprint_radius_label)
        footprint_layout.addWidget(self.footprint_radius_input)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(size_layout)
        main_layout.addLayout(footprint_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def median_size_changed(self):
        pass


    def footprint_radius_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        size_xy = int(self.median_size_xy_input.text().strip())
        size_z = int(self.median_size_z_input.text().strip())
        footprint = None
        footprint_text = self.footprint_combo_box.currentText()
        footprint_radius = int(self.footprint_radius_input.text().strip())
        if not footprint_text == "none":
            if footprint_text == "cube":
                footprint_width = 2 * footprint_radius + 1
                footprint = skimage.morphology.footprint_rectangle((footprint_width, footprint_width, footprint_width))
            else:
                footprint_function = getattr(skimage.morphology, footprint_text)
                footprint = footprint_function(footprint_radius)
        mode = self.mode_combo_box.currentText()
        self.filter = MedianFilter(self.input_layer.data)
        self.filter.size = (size_z, size_xy, size_xy)
        self.filter.footprint = footprint
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying median filter...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " median"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )


# noinspection PyTypeChecker
class GaussianFilterWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.setWindowTitle("Gaussian Filter")
        self.viewer = viewer
        self.sigma_xy = 1.5
        self.sigma_z = 0.5
        self.sigma_xy_input = None
        self.sigma_z_input = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        sigma_xy_label, self.sigma_xy_input = WidgetTool.getLineInput(self, "sigma xy:",
                                                            self.sigma_xy,
                                                            self.field_width,
                                                            self.sigma_changed)
        sigma_z_label, self.sigma_z_input = WidgetTool.getLineInput(self, "sigma z:",
                                                                   self.sigma_z,
                                                                   self.field_width,
                                                                   self.sigma_changed)
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                                 self.modes,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        size_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        size_layout.addWidget(sigma_xy_label)
        size_layout.addWidget(self.sigma_xy_input)
        size_layout.addWidget(sigma_z_label)
        size_layout.addWidget(self.sigma_z_input)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(size_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def sigma_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        sigma_xy = float(self.sigma_xy_input.text().strip())
        sigma_z = float(self.sigma_z_input.text().strip())
        mode = self.mode_combo_box.currentText()
        self.filter = GaussianFilter(self.input_layer.data)
        self.filter.sigma = (sigma_z, sigma_xy, sigma_xy)
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying median filter...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " gaussian"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )


class AnisotropicDiffusionFilterWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.setWindowTitle("Anisotropic Diffusion Filter")
        self.viewer = viewer
        self.iterations = 5
        self.kappa = 50
        self.gamma = 0.1
        self.step_xy = 1.
        self.step_z = 1.
        self.option = 1
        self.iterations_input = None
        self.kappa_input = None
        self.gamma_input = None
        self.step_xy_input = None
        self.step_z_input = None
        self.options = ["favour high contrast edges", "favour wide regions"]
        self.options_combo_box = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        iterations_label, self.iterations_input = WidgetTool.getLineInput(self, "iterations:",
                                                            self.iterations,
                                                            self.field_width,
                                                            self.iterations_changed)
        kappa_label, self.kappa_input = WidgetTool.getLineInput(self, "kappa:",
                                                            self.kappa,
                                                            self.field_width,
                                                            self.kappa_changed)
        gamma_label, self.gamma_input = WidgetTool.getLineInput(self, "gamma:",
                                                          self.gamma,
                                                          self.field_width,
                                                          self.gamma_changed)
        step_xy_label, self.step_xy_input = WidgetTool.getLineInput(self, "step xy:",
                                                          self.step_xy,
                                                          self.field_width,
                                                          self.step_changed)
        step_z_label, self.step_z_input = WidgetTool.getLineInput(self, "step z:",
                                                                    self.step_z,
                                                                    self.field_width,
                                                                    self.step_changed)
        option_label, self.options_combo_box = WidgetTool.getComboInput(self, "equation:",
                                                                                 self.options,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        iterations_layout = QHBoxLayout()
        kappa_gamma_layout = QHBoxLayout()
        step_layout = QHBoxLayout()
        option_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations_input)
        kappa_gamma_layout.addWidget(kappa_label)
        kappa_gamma_layout.addWidget(self.kappa_input)
        kappa_gamma_layout.addWidget(gamma_label)
        kappa_gamma_layout.addWidget(self.gamma_input)
        step_layout.addWidget(step_xy_label)
        step_layout.addWidget(self.step_xy_input)
        step_layout.addWidget(step_z_label)
        step_layout.addWidget(self.step_z_input)
        option_layout.addWidget(option_label)
        option_layout.addWidget(self.options_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(iterations_layout)
        main_layout.addLayout(kappa_gamma_layout)
        main_layout.addLayout(step_layout)
        main_layout.addLayout(option_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def iterations_changed(self):
        pass


    def kappa_changed(self):
        pass


    def gamma_changed(self):
        pass


    def step_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        iterations = int(self.iterations_input.text().strip())
        kappa = float(self.kappa_input.text().strip())
        gamma = float(self.gamma_input.text().strip())
        step_xy = float(self.step_xy_input.text().strip())
        step_z = float(self.step_z_input.text().strip())
        option = self.options_combo_box.currentIndex() + 1
        self.filter = AnisotropicDiffusionFilter(self.input_layer.data)
        self.filter.iterations = iterations
        self.filter.kappa = kappa
        self.filter.gamma = gamma
        self.filter.step = (step_z, step_xy, step_xy)
        self.filter.option = option
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying anisotropic diffusion filter...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " anisodiff"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class RollingBallWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.radius = 50
        self.radius_input = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        radius_label, self.radius_input = WidgetTool.getLineInput(self, "radius:",
                                                            self.radius,
                                                            self.field_width,
                                                            self.radius_changed)
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        radius_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_input)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(radius_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def radius_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        radius = int(self.radius_input.text().strip())
        self.filter = RollingBall(self.input_layer.data)
        self.filter.radius = radius
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying background subtraction...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " background"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )
        


class ThresholdWidget(ToolboxWidget):
    
    
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.min_value = 0
        self.max_value = 255
        self.min_value_slider = None
        self.max_value_slider = None
        self.min_value_input = None
        self.max_value_input = None
        self.original_cmap = None
        self.original_blending = None
        self.current_layer = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)
        self.update_current_layer()


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        self.input_layer_combo_box.currentIndexChanged.connect(self.update_current_layer)
        self.min_value_slider = QSlider(Qt.Horizontal)
        self.min_value_slider.valueChanged.connect(self.min_threshold_changed)
        min_value_label, self.min_value_input = WidgetTool.getLineInput(self, "min.:",
                                                                        self.min_value,
                                                                        self.field_width,
                                                                        self.min_value_input_changed
                                                                        )
        self.min_value_input.textChanged.connect(self.min_value_input_changed)
        self.max_value_slider = QSlider(Qt.Horizontal)
        self.max_value_slider.valueChanged.connect(self.max_threshold_changed)
        max_value_label, self.max_value_input = WidgetTool.getLineInput(self, "max.:",
                                                                        self.max_value,
                                                                        self.field_width,
                                                                        self.max_value_input_changed
                                                                        )

        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        min_layout = QHBoxLayout()
        max_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        min_layout.addWidget(min_value_label)
        min_layout.addWidget(self.min_value_slider)
        min_layout.addWidget(self.min_value_input)
        max_layout.addWidget(max_value_label)
        max_layout.addWidget(self.max_value_slider)
        max_layout.addWidget(self.max_value_input)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(min_layout)
        main_layout.addLayout(max_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)


    def update_current_layer(self):
        new_layer = self.napari_util.getLayerWithName(self.input_layer_combo_box.currentText())
        if not new_layer or not isinstance(new_layer, Image) or new_layer is self.current_layer:
            return
        if self.current_layer:
            self.current_layer.colormap = self.original_cmap
            self.current_layer.blending = self.original_blending
        self.current_layer = new_layer
        self.original_cmap = new_layer.colormap
        self.original_blending = new_layer.blending
        new_layer.colormap = 'HiLo'
        new_layer.blending = "additive"
        sliders_min = int(round(new_layer.contrast_limits_range[0]))
        sliders_max = int(round(new_layer.contrast_limits_range[1]))
        if new_layer.data.dtype in (np.uint8, np.uint16, uint32):
            sliders_min = np.iinfo(new_layer.data.dtype).min
            sliders_max = np.iinfo(new_layer.data.dtype).max
        self.min_value_slider.setMinimum(sliders_min)
        self.min_value_slider.setMaximum(sliders_max)
        self.min_value_slider.setValue(sliders_min)
        self.max_value_slider.setMinimum(sliders_min)
        self.max_value_slider.setMaximum(sliders_max)
        self.max_value_slider.setValue(sliders_max)
        if 'float' in str(new_layer.data.dtype):
            self.min_value_slider.setMinimum(0)
            self.min_value_slider.setMaximum(65535)
            self.min_value_slider.setValue(0)
            self.max_value_slider.setMinimum(0)
            self.max_value_slider.setMaximum(65535)
            self.max_value_slider.setValue(65535)



    def min_threshold_changed(self, value):
        new_value = value
        max_threshold_value = self.max_value_slider.value()
        max_threshold_limit = self.max_value_slider.maximum()
        if value >= max_threshold_value and max_threshold_value < max_threshold_limit:
            self.max_value_slider.setValue(value + 1)
        if value >= max_threshold_value == max_threshold_limit:
            new_value = value - 1
        text = self.input_layer_combo_box.currentText()
        layer = self.napari_util.getLayerWithName(text)
        data_min = np.min(layer.data)
        data_max = np.max(layer.data)
        new_value_float = new_value
        if 'float' in str(layer.data.dtype):
            new_value_float = data_min + ((new_value/65535.0) * (data_max - data_min))
            max_threshold_value = data_min + ((max_threshold_value/65535.0) * (data_max - data_min))
        self.current_layer.contrast_limits = [new_value_float, max_threshold_value]
        self.min_value_input.setText(str(new_value))


    def max_threshold_changed(self, value):
        new_value = value
        min_threshold_value = self.min_value_slider.value()
        min_threshold_limit = self.max_value_slider.minimum()
        if value <= min_threshold_value and min_threshold_value > min_threshold_limit:
            self.min_value_slider.setValue(value - 1)
        if value <= min_threshold_value == min_threshold_limit:
            new_value = value + 1
        text = self.input_layer_combo_box.currentText()
        layer = self.napari_util.getLayerWithName(text)
        data_min = np.min(layer.data)
        data_max = np.max(layer.data)
        new_value_float = new_value
        if 'float' in str(layer.data.dtype):
            new_value_float = data_min + ((new_value/65535.0) * (data_max - data_min))
            min_threshold_value = data_min + ((min_threshold_value/65535.0) * (data_max - data_min))
        self.current_layer.contrast_limits = [min_threshold_value, new_value_float]
        self.max_value_input.setText(str(new_value))


    def min_value_input_changed(self, value):
        number = str_to_number(value)
        self.min_value_slider.setValue(number)


    def max_value_input_changed(self, value):
        number = str_to_number(value)
        self.max_value_slider.setValue(number)


    def on_apply_button_clicked(self):
        self.input_layer = self.current_layer
        data_min = np.min(self.input_layer.data)
        data_max = np.max(self.input_layer.data)
        min_value = self.min_value_slider.value()
        max_value = self.max_value_slider.value()
        if 'float' in str(self.input_layer.data.dtype):
            min_value = data_min + ((min_value / 65535.0) * (data_max - data_min))
            max_value = data_min + ((max_value / 65535.0) * (data_max - data_min))
        self.filter = Threshold(self.current_layer.data)
        self.filter.min_value = min_value
        self.filter.max_value = max_value
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Thresholding image...'}
                               )
        worker.finished.connect(self.on_operation_finished)
        worker.start()


    def on_operation_finished(self):
        name = self.input_layer.name + " mask"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )


class EdgeFilterWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.sigmas = [1, 3]
        self.black_ridges = False
        self.sigmas_input = None
        self.black_ridges_checkbox = None


    def get_sigmas_as_text(self):
        return ','.join([str(sigma) for sigma in self.sigmas])



class FrangiFilterWidget(EdgeFilterWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = None
        self.alpha_input = None
        self.beta_input = None
        self.gamma_input = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        sigmas_label, self.sigmas_input = WidgetTool.getLineInput(self, "sigmas:",
                                                                                  self.get_sigmas_as_text(),
                                                                                  self.field_width,
                                                                                  self.sigmas_changed)
        alpha_label, self.alpha_input = WidgetTool.getLineInput(self, "alpha:",
                                                                                  self.alpha,
                                                                                  self.field_width,
                                                                                  self.alpha_changed)
        beta_label, self.beta_input = WidgetTool.getLineInput(self, "beta:",
                                                               self.beta,
                                                               self.field_width,
                                                               self.beta_changed)
        gamma_label, self.gamma_input = WidgetTool.getLineInput(self, "gamma:",
                                                             self.gamma,
                                                             self.field_width,
                                                             self.gamma_changed)
        self.black_ridges_checkbox = QCheckBox("black ridges")
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                                 self.modes,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        sigma_layout = QHBoxLayout()
        abc_layout = QHBoxLayout()
        black_ridges_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        sigma_layout.addWidget(sigmas_label)
        sigma_layout.addWidget(self.sigmas_input)
        abc_layout.addWidget(alpha_label)
        abc_layout.addWidget(self.alpha_input)
        abc_layout.addWidget(beta_label)
        abc_layout.addWidget(self.beta_input)
        abc_layout.addWidget(gamma_label)
        abc_layout.addWidget(self.gamma_input)
        black_ridges_layout.addWidget(self.black_ridges_checkbox)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(sigma_layout)
        main_layout.addLayout(abc_layout)
        main_layout.addLayout(black_ridges_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def sigmas_changed(self, value):
        self.sigmas = value.strip().split(",")
        self.sigmas = [float(sigma.strip()) for sigma in self.sigmas]


    def alpha_changed(self):
        pass


    def beta_changed(self):
        pass


    def gamma_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        alpha = float(self.alpha_input.text().strip())
        beta = float(self.beta_input.text().strip())
        gamma_text = self.gamma_input.text().strip()
        gamma = None
        if not gamma_text in ['NONE', "None", "none"]:
            gamma = float(gamma_text)
        black_ridges = self.black_ridges_checkbox.isChecked()
        mode = self.mode_combo_box.currentText()
        self.filter = FrangiFilter(self.input_layer.data)
        self.filter.sigmas = self.sigmas
        self.filter.alpha = alpha
        self.filter.beta = beta
        self.filter.gamma = gamma
        self.filter.black_ridges = black_ridges
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying Frangi Filter...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " frangi"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive',
            colormap='inferno'
        )



class SatoFilterWidget(EdgeFilterWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        sigmas_label, self.sigmas_input = WidgetTool.getLineInput(self, "sigmas:",
                                                                                  self.get_sigmas_as_text(),
                                                                                  self.field_width,
                                                                                  self.sigmas_changed)
        self.black_ridges_checkbox = QCheckBox("black ridges")
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                                 self.modes,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        sigma_layout = QHBoxLayout()
        black_ridges_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        sigma_layout.addWidget(sigmas_label)
        sigma_layout.addWidget(self.sigmas_input)
        black_ridges_layout.addWidget(self.black_ridges_checkbox)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(sigma_layout)
        main_layout.addLayout(black_ridges_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def sigmas_changed(self, value):
        self.sigmas = value.strip().split(",")
        self.sigmas = [float(sigma.strip()) for sigma in self.sigmas]


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        black_ridges = self.black_ridges_checkbox.isChecked()
        mode = self.mode_combo_box.currentText()
        self.filter = SatoFilter(self.input_layer.data)
        self.filter.sigmas = self.sigmas
        self.filter.black_ridges = black_ridges
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying Sato Filter...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " sato"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive',
            colormap='inferno'
        )


class MeijeringFilterWidget(EdgeFilterWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.alpha = None
        self.alpha_input = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        sigmas_label, self.sigmas_input = WidgetTool.getLineInput(self, "sigmas:",
                                                                                  self.get_sigmas_as_text(),
                                                                                  self.field_width,
                                                                                  self.sigmas_changed)
        alpha_label, self.alpha_input = WidgetTool.getLineInput(self, "alpha:",
                                                                                  self.alpha,
                                                                                  self.field_width,
                                                                                  self.alpha_changed)
        self.black_ridges_checkbox = QCheckBox("black ridges")
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                                 self.modes,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        sigma_layout = QHBoxLayout()
        abc_layout = QHBoxLayout()
        black_ridges_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        sigma_layout.addWidget(sigmas_label)
        sigma_layout.addWidget(self.sigmas_input)
        abc_layout.addWidget(alpha_label)
        abc_layout.addWidget(self.alpha_input)
        black_ridges_layout.addWidget(self.black_ridges_checkbox)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(sigma_layout)
        main_layout.addLayout(abc_layout)
        main_layout.addLayout(black_ridges_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def sigmas_changed(self, value):
        self.sigmas = value.strip().split(",")
        self.sigmas = [float(sigma.strip()) for sigma in self.sigmas]


    def alpha_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        alpha_text = self.alpha_input.text().strip()
        alpha = None
        if not alpha_text in ['NONE', "None", "none"]:
            alpha = float(alpha_text)
        black_ridges = self.black_ridges_checkbox.isChecked()
        mode = self.mode_combo_box.currentText()
        self.filter = MeijeringFilter(self.input_layer.data)
        self.filter.sigmas = self.sigmas
        self.filter.alpha = alpha
        self.filter.black_ridges = black_ridges
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying Meijering Filter...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " meijering"
        self.viewer.add_image(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive',
            colormap='inferno'
        )



class DilationWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.footprint_radius = 1
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def get_footprints(self):
        return self.footprints[1:]


    def get_modes(self):
        return ['ignore', 'min', 'max']


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        footprint_label, self.footprint_combo_box = WidgetTool.getComboInput(self, "footprint:",
                                                                             self.get_footprints(),
                                                                             )
        footprint_radius_label, self.footprint_radius_input = WidgetTool.getLineInput(self, "radius:",
                                                                                      self.footprint_radius,
                                                                                      self.field_width,
                                                                                      self.footprint_radius_changed)
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                   self.get_modes(),
                                                                   )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        footprint_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        footprint_layout.addWidget(footprint_label)
        footprint_layout.addWidget(self.footprint_combo_box)
        footprint_layout.addWidget(footprint_radius_label)
        footprint_layout.addWidget(self.footprint_radius_input)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(footprint_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def footprint_radius_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        footprint_text = self.footprint_combo_box.currentText()
        footprint_radius = int(self.footprint_radius_input.text().strip())
        print('footprint_radius', footprint_radius)
        footprint = self.get_footprint(footprint_text, footprint_radius,  self.input_layer.data.ndim)
        mode = self.mode_combo_box.currentText()
        self.filter = Dilation(self.input_layer.data)
        self.filter.footprint = footprint
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying dilation...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " dilation"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class ClosingWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.footprint_radius = 1
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def get_footprints(self):
        return self.footprints[1:]


    def get_modes(self):
        return ['ignore', 'min', 'max']


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        footprint_label, self.footprint_combo_box = WidgetTool.getComboInput(self, "footprint:",
                                                                             self.get_footprints(),
                                                                             )
        footprint_radius_label, self.footprint_radius_input = WidgetTool.getLineInput(self, "radius:",
                                                                                      self.footprint_radius,
                                                                                      self.field_width,
                                                                                      self.footprint_radius_changed)
        mode_label, self.mode_combo_box = WidgetTool.getComboInput(self, "mode:",
                                                                   self.get_modes(),
                                                                   )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        footprint_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        footprint_layout.addWidget(footprint_label)
        footprint_layout.addWidget(self.footprint_combo_box)
        footprint_layout.addWidget(footprint_radius_label)
        footprint_layout.addWidget(self.footprint_radius_input)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(footprint_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def footprint_radius_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        footprint_text = self.footprint_combo_box.currentText()
        footprint_radius = int(self.footprint_radius_input.text().strip())
        footprint = self.get_footprint(footprint_text, footprint_radius, self.input_layer.data.ndim)
        mode = self.mode_combo_box.currentText()
        self.filter = Closing(self.input_layer.data)
        self.filter.footprint = footprint
        self.filter.mode = mode
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Applying closing...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " close"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class LabelWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.connectivities = ['1', '2', '3']
        self.connectivity_combo_box = None
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        self.label_layer_combo_box.currentIndexChanged.connect(self.on_layer_changed)
        connectivity_label, self.connectivity_combo_box = WidgetTool.getComboInput(self, "connectivity:",
                                                                             self.connectivities,
                                                                             )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        connectivity_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        connectivity_layout.addWidget(connectivity_label)
        connectivity_layout.addWidget(self.connectivity_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(connectivity_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def on_layer_changed(self):
        layer = self.napari_util.getLayerWithName(self.label_layer_combo_box.currentText())
        if not isinstance(layer, Labels):
            return
        self.connectivity_combo_box.setCurrentText(str(layer.data.ndim))


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        connectivity = int(self.connectivity_combo_box.currentText())
        if self.input_layer.data.ndim == 2 and connectivity == 3:
            connectivity = 2
        self.filter = Label(self.input_layer.data)
        self.filter.connectivity = connectivity
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Labeling mask...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " labels"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class RemoveSmallObjectsWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.min_size = 64
        self.min_size_input = None
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        min_size_label, self.min_size_input = WidgetTool.getLineInput(self, "min. size:",
                                                                                      self.min_size,
                                                                                      self.field_width,
                                                                                      self.min_size_changed)
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        min_size_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        min_size_layout.addWidget(min_size_label)
        min_size_layout.addWidget(self.min_size_input)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(min_size_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def min_size_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        min_size = int(self.min_size_input.text().strip())
        self.filter = RemoveSmallObjects(self.input_layer.data)
        self.filter.min_size = min_size
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Removing small objects...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " small objects removed"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class ClearBorderWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        self.filter = ClearBorder(self.input_layer.data)
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Clearing the border...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " cleared border"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class SkeletonizeWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.method = "lee"
        self.methods = ["lee", "zhang"]
        self.method_combo_box = None
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        self.label_layer_combo_box.currentIndexChanged.connect(self.on_layer_changed)
        method_label, self.method_combo_box = WidgetTool.getComboInput(self, "method:",
                                                                                   self.methods,
                                                                                   )
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        method_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo_box)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(method_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def on_layer_changed(self):
        layer = self.napari_util.getLayerWithName(self.label_layer_combo_box.currentText())
        if not isinstance(layer, Labels):
            return
        if layer.data.ndim == 3 and self.method_combo_box.currentText() == "zhang":
            self.method_combo_box.setCurrentText("lee")


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        self.filter = Skeletonize(self.input_layer.data)
        self.filter.method = self.method_combo_box.currentText()
        if self.input_layer.data.ndim == 3 and  self.filter.method == "zhang":
            self.filter.method = 'lee'
            notifications.show_info("Can't apply zhang to 3D data, using lee instead.")
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Skeletonizing ('+self.filter.method+')...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " skeleton-" + self.filter.method
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class HamiltonJacobiSkeletonizeWidget(ToolboxWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.flux_threshold = 2.5  # gamma
        self.dilation = 1.5        # epsilon
        self.use_anisotropic_diffusion = False
        self.flux_threshold_input = None
        self.dilation_input = None
        self.use_anisotropic_diffusion_checkbox = None
        self.create_layout()
        self.label_combo_boxes.append(self.label_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.label_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.label_layers,
                                                                                 )
        flux_threshold_label, self.flux_threshold_input = WidgetTool.getLineInput(self, "flux threshold:",
                                                                                      self.flux_threshold,
                                                                                      self.field_width,
                                                                                      self.flux_threshold_changed)
        dilation_label, self.dilation_input = WidgetTool.getLineInput(self, "dilation:",
                                                                                  self.dilation,
                                                                                  self.field_width,
                                                                                  self.dilation_changed)
        self.use_anisotropic_diffusion_checkbox = QCheckBox("use anisotropic diffusion")
        apply_button = QPushButton("&Apply")
        apply_button.clicked.connect(self.on_apply_button_clicked)
        layer_layout = QHBoxLayout()
        flux_layout = QHBoxLayout()
        dilation_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.label_layer_combo_box)
        flux_layout.addWidget(flux_threshold_label)
        flux_layout.addWidget(self.flux_threshold_input)
        dilation_layout.addWidget(dilation_label)
        dilation_layout.addWidget(self.dilation_input)
        button_layout.addWidget(apply_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(flux_layout)
        main_layout.addLayout(dilation_layout)
        main_layout.addWidget(self.use_anisotropic_diffusion_checkbox)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def flux_threshold_changed(self):
        pass


    def dilation_changed(self):
        pass


    def on_apply_button_clicked(self):
        text = self.label_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        flux_threshold = float(self.flux_threshold_input.text().strip())
        dilation_parameter = float(self.dilation_input.text().strip())
        use_anisotropic_diffusion = self.use_anisotropic_diffusion_checkbox.isChecked()
        self.filter = HamiltonJacobiSkeleton(self.input_layer.data)
        self.filter.flux_threshold = flux_threshold
        self.filter.dilation = dilation_parameter
        self.filter.use_anisotropic_diffusion = use_anisotropic_diffusion
        worker = create_worker(self.filter.run,
                               _progress={'desc': 'Hamilton-Jacobi Skeletonizing...'}
                               )
        worker.finished.connect(self.on_filter_finished)
        worker.start()


    def on_filter_finished(self):
        name = self.input_layer.name + " HJS"
        self.viewer.add_labels(
            self.filter.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )



class PixelClassifierWidget(ToolboxWidget):
    
    
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.point_layers = self.napari_util.getPointsLayers()
        self.point_layer_combo_box = None
        self.intensity_features = True
        self.edges_features = True
        self.texture_features = True
        self.intensity_check_box = None
        self.edges_check_box = None
        self.texture_check_box = None
        self.sigma_min = 1
        self.sigma_max = 16
        self.num_sigma = None
        self.sigma_min_input = None
        self.sigma_max_input = None
        self.num_sigma_input = None
        self.pixelClassifier = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)
        self.point_combo_boxes.append(self.point_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.image_layers,
                                                                                 )
        point_layer_label, self.point_layer_combo_box = WidgetTool.getComboInput(self, "points:",
                                                                                 self.point_layers,
                                                                                 )
        self.intensity_check_box = QCheckBox("Intensity features")
        self.intensity_check_box.setChecked(self.intensity_features)
        self.edges_check_box = QCheckBox("Edges Features")
        self.edges_check_box.setChecked(self.edges_features)
        self.texture_check_box = QCheckBox("Texture Features")
        self.texture_check_box.setChecked(self.texture_features)
        sigma_min_label, self.sigma_min_input = WidgetTool.getLineInput(self, "sigma min.:",
                                                                                  self.sigma_min,
                                                                                  self.field_width,
                                                                                  self.sigma_changed)
        sigma_max_label, self.sigma_max_input = WidgetTool.getLineInput(self, "sigma max.:",
                                                                        self.sigma_max,
                                                                        self.field_width,
                                                                        self.sigma_changed)
        num_sigma_label, self.num_sigma_input = WidgetTool.getLineInput(self, "num. sigma:",
                                                                        self.num_sigma,
                                                                        self.field_width,
                                                                        self.sigma_changed)
        train_button = QPushButton("&Train")
        train_button.clicked.connect(self.on_train_button_clicked)
        classify_button = QPushButton("&Classify")
        classify_button.clicked.connect(self.on_classify_button_clicked)

        layer_layout = QHBoxLayout()
        point_layout = QHBoxLayout()
        checkboxes_layout = QVBoxLayout()
        sigma_min_layout = QHBoxLayout()
        sigma_max_layout = QHBoxLayout()
        num_sigma_layout = QHBoxLayout()
        button_layout = QVBoxLayout()

        layer_layout.addWidget(input_layer_label)
        layer_layout.addWidget(self.input_layer_combo_box)
        point_layout.addWidget(point_layer_label)
        point_layout.addWidget(self.point_layer_combo_box)
        checkboxes_layout.addWidget(self.intensity_check_box)
        checkboxes_layout.addWidget(self.edges_check_box)
        checkboxes_layout.addWidget(self.texture_check_box)
        sigma_min_layout.addWidget(sigma_min_label)
        sigma_min_layout.addWidget(self.sigma_min_input)
        sigma_max_layout.addWidget(sigma_max_label)
        sigma_max_layout.addWidget(self.sigma_max_input)
        num_sigma_layout.addWidget(num_sigma_label)
        num_sigma_layout.addWidget(self.num_sigma_input)
        button_layout.addWidget(train_button)
        button_layout.addWidget(classify_button)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(point_layout)
        main_layout.addLayout(checkboxes_layout)
        main_layout.addLayout(sigma_min_layout)
        main_layout.addLayout(sigma_max_layout)
        main_layout.addLayout(num_sigma_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def sigma_changed(self):
        pass


    def on_train_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        text = self.point_layer_combo_box.currentText()
        point_layer = self.napari_util.getLayerWithName(text)
        use_intensity = self.intensity_check_box.isChecked()
        use_edges = self.edges_check_box.isChecked()
        use_texture = self.texture_check_box.isChecked()
        sigma_min = int(self.sigma_min_input.text().strip())
        sigma_max = int(self.sigma_max_input.text().strip())
        num_sigma = self.sigma_max_input.text().strip()
        if num_sigma in ['None', 'NONE', 'none']:
            num_sigma = None
        else:
            num_sigma = int(num_sigma)
        self.pixelClassifier = RandomForestPixelClassifier(self.input_layer.data)
        self.pixelClassifier.training_points = point_layer.data
        self.pixelClassifier.training_points_classes = [str(e) for e in point_layer.face_color]
        self.pixelClassifier.intensity = use_intensity
        self.pixelClassifier.edges = use_edges
        self.pixelClassifier.texture = use_texture
        self.pixelClassifier.sigma_min = sigma_min
        self.pixelClassifier.sigma_max = sigma_max
        self.pixelClassifier.num_sigma = num_sigma
        worker = create_worker(self.pixelClassifier.train,
                               _progress={'desc': 'Training Pixel Classifier...'}
                               )
        worker.finished.connect(self.on_train_finished)
        worker.start()


    def on_train_finished(self):
        self.pixelClassifier.predict()
        name = self.input_layer.name + " labels"
        self.viewer.add_labels(
            self.pixelClassifier.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )


    def on_classify_button_clicked(self):
        text = self.input_layer_combo_box.currentText()
        self.input_layer = self.napari_util.getLayerWithName(text)
        self.pixelClassifier.image = self.input_layer.data
        self.pixelClassifier.predict()
        name = self.input_layer.name + " labels"
        self.viewer.add_labels(
            self.pixelClassifier.result,
            name=name,
            scale=self.input_layer.scale,
            units=self.input_layer.units,
            blending='additive'
        )