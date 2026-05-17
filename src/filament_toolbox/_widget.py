"""
This module contains the tools of the filament-toolbox
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import skimage.morphology
from autooptions import Options
from autooptions import OptionsWidget
from napari.layers import Image
from napari.layers import Labels
from napari.qt.threading import create_worker
from napari.utils.events import Event
from numba.core.types import uint32
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QSlider
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget
from skimage.color import rgb2gray

from filament_toolbox.lib.filter import AnisotropicDiffusionFilter
from filament_toolbox.lib.filter import FrangiFilter
from filament_toolbox.lib.filter import GaussianFilter
from filament_toolbox.lib.filter import MedianFilter
from filament_toolbox.lib.filter import MeijeringFilter
from filament_toolbox.lib.filter import RollingBall
from filament_toolbox.lib.filter import SatoFilter
from filament_toolbox.lib.measure import MeasureLabels
from filament_toolbox.lib.measure import MeasureSkeleton
from filament_toolbox.lib.metric import CenterlineDice
from filament_toolbox.lib.metric import Dice
from filament_toolbox.lib.ml import RandomForestPixelClassifier
from filament_toolbox.lib.morphology import Closing
from filament_toolbox.lib.morphology import Dilation
from filament_toolbox.lib.morphology import EuclideanDistanceTransform
from filament_toolbox.lib.morphology import HamiltonJacobiSkeleton
from filament_toolbox.lib.morphology import Label
from filament_toolbox.lib.morphology import LocalThickness
from filament_toolbox.lib.morphology import RemoveSmallObjects
from filament_toolbox.lib.morphology import Skeletonize
from filament_toolbox.lib.napari_util import NapariUtil
from filament_toolbox.lib.qtutil import TableView
from filament_toolbox.lib.qtutil import WidgetTool
from filament_toolbox.lib.segmentation import ClearBorder
from filament_toolbox.lib.segmentation import Threshold
from filament_toolbox.lib.tracing import BrightestPathTracing
from filament_toolbox.lib.transform import IsotropicResampling

if TYPE_CHECKING:
    import napari


def activate():
    print("Filament Toolbox activated")


def rgb_to_8bit(viewer: "napari.viewer.Viewer"):
    layer = viewer.layers.selection.active
    name = layer.name + " 8bit"
    converted = rgb2gray(layer.data)
    converted = (converted * 255).astype(np.uint8)
    viewer.add_image(
        converted,
        name=name,
        scale=layer.scale,
        units=layer.units,
        blending="additive",
        colormap="gray",
    )


def rgb_to_16bit(viewer: "napari.viewer.Viewer"):
    layer = viewer.layers.selection.active
    name = layer.name + " 16bit"
    converted = rgb2gray(layer.data)
    converted = (converted * 65535).astype(np.uint16)
    viewer.add_image(
        converted,
        name=name,
        scale=layer.scale,
        units=layer.units,
        blending="additive",
        colormap="gray",
    )


def str_to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


class SimpleWidget(QWidget):

    def __init__(self, viewer, sameRowSet=None):
        super().__init__()
        self.modes = ["reflect", "constant", "nearest", "mirror", "wrap"]
        self.viewer = viewer
        self.options = self.getOptions()
        self.widget = None
        self.sameRowSet = sameRowSet
        self.operation = None
        self.imageLayer = None
        self.createLayout()

    def addModesOption(self, options):
        options.addChoice("mode", choices=self.modes, value=self.modes[0])

    def createLayout(self):
        self.widget = OptionsWidget(
            self.viewer,
            self.options,
            client=self,
            layout_type="vertical",
            sameRowSet=self.sameRowSet,
        )
        self.widget.addApplyButton(self.apply)
        layout = QVBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def displayImage(self, name, colormap=None):
        self.viewer.add_image(
            self.operation.result,
            name=name,
            scale=self.imageLayer.scale,
            units=self.imageLayer.units,
            blending="additive",
            colormap=colormap,
        )

    def displayLabels(self, name):
        self.viewer.add_labels(
            self.operation.result,
            name=name,
            scale=self.imageLayer.scale,
            units=self.imageLayer.units,
            blending="additive",
        )

    def sigmasChanged(self, value):
        pass  # default implementation does nothing

    @abstractmethod
    def getOptions(self):
        raise Exception(
            "Abstract method getOptions of class SimpleWidget called!"
        )

    @abstractmethod
    def apply(self):
        raise Exception("Abstract method apply of class SimpleWidget called!")

    @abstractmethod
    def displayResult(self):
        raise Exception(
            "Abstract method displayResult of class SimpleWidget called!"
        )

    def runOperationInThread(self, description, callback=None):
        worker = create_worker(
            self.operation.run, _progress={"desc": description}
        )
        if callback is not None:
            worker.finished.connect(callback)
        worker.start()


class MorphologySimpleWidget(SimpleWidget):

    def __init__(self, viewer, sameRowSet=None):
        super().__init__(viewer, sameRowSet)

    def addFootprintOptions(self, options, radius=1, withNone=False):
        choices = ["cube", "ball", "octahedron"]
        if withNone:
            choices = ["none"] + choices
        options.addChoice("footprint", choices=choices, value=choices[0])
        options.addInt("radius", 1)
        options.addChoice(
            "mode",
            choices=self.modes + ["ignore", "min", "max"],
            value="ignore",
        )

    @classmethod
    def getFootprint(cls, name, radius, dims):
        se_name = name
        two_d_ses = {"ball": "disk", "octahedron": "diamond"}
        if dims == 2:
            if name in two_d_ses.keys():
                se_name = two_d_ses[name]
        if name == "cube":
            footprint_width = 2 * radius + 1
            if dims == 2:
                footprint = skimage.morphology.footprint_rectangle(
                    (footprint_width, footprint_width)
                )
            else:
                footprint = skimage.morphology.footprint_rectangle(
                    (footprint_width, footprint_width, footprint_width)
                )
        else:
            footprint_function = getattr(skimage.morphology, se_name)
            footprint = footprint_function(radius)
        return footprint

    @abstractmethod
    def apply(self):
        raise Exception("Abstract method apply of class SimpleWidget called!")

    @abstractmethod
    def displayResult(self):
        raise Exception(
            "Abstract method displayResult of class SimpleWidget called!"
        )

    @abstractmethod
    def getOptions(self):
        raise Exception(
            "Abstract method getOptions of class SimpleWidget called!"
        )


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
        self.viewer.layers.events.inserted.connect(
            self.on_layer_added_or_removed
        )
        self.viewer.layers.events.removed.connect(
            self.on_layer_added_or_removed
        )

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
        two_d_ses = {"ball": "disk", "octahedron": "diamond"}
        if dims == 2:
            if name in two_d_ses.keys():
                se_name = two_d_ses[name]
        if name == "cube":
            footprint_width = 2 * radius + 1
            if dims == 2:
                footprint = skimage.morphology.footprint_rectangle(
                    (footprint_width, footprint_width)
                )
            else:
                footprint = skimage.morphology.footprint_rectangle(
                    (footprint_width, footprint_width, footprint_width)
                )
        else:
            footprint_function = getattr(skimage.morphology, se_name)
            footprint = footprint_function(radius)
        return footprint


class AnisotropicDiffusionFilterWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament Toolbox", "anisotropic_diffusion_filter")
        options.addImage()
        options.addInt("iterations", value=5)
        options.addInt("kappa", value=50)
        options.addFloat("gamma", value=0.1)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = AnisotropicDiffusionFilter(self.imageLayer.data)
        self.operation.iterations = self.options.value("iterations")
        self.operation.kappa = self.options.value("kappa")
        self.operation.gamma = self.options.value("gamma")
        steps = self.imageLayer.scale
        if len(steps) < 3:
            steps = (1, steps[0], steps[1])
        self.operation.step = steps
        self.runOperationInThread(
            "Applying Anisotropic Diffusion Filter...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " anisodiff"
        self.displayImage(name)


class MeasureSkeletonWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament Toolbox", "measure_skeleton")
        options.addImage()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = MeasureSkeleton(self.imageLayer.data)
        self.operation.scale = self.imageLayer.scale
        self.operation.units = self.imageLayer.units
        self.runOperationInThread(
            "Measuring Skeleton...", callback=self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " skeleton"
        self.displayLabels(name)
        table = TableView(self.operation.table)
        self.viewer.window.add_dock_widget(
            table, name="skeleton measurements", tabify=True, area="right"
        )


# noinspection PyTypeChecker
class GaussianFilterWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer, sameRowSet={"sigma z"})

    def getOptions(self):
        options = Options("Filament Toolbox", "gaussian_filter")
        options.addImage()
        options.addFloat("sigma xy", value=1.5)
        options.addFloat("sigma z", value=0.5)
        options.addChoice("mode", choices=self.modes, value=self.modes[0])
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = GaussianFilter(self.imageLayer.data)
        self.operation.sigma = (
            self.options.value("sigma z"),
            self.options.value("sigma xy"),
            self.options.value("sigma xy"),
        )
        self.operation.mode = self.options.value("mode")
        self.runOperationInThread(
            "Applying Gaussian...", callback=self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " Gaussian"
        self.displayImage(name)


class IsotropicResamplingWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament Toolbox", "isotropic_resampling")
        options.addImage()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        scale = self.imageLayer.scale
        if isinstance(scale, np.ndarray):
            scale = scale.tolist()
        self.operation = IsotropicResampling(self.imageLayer.data, scale)
        self.runOperationInThread(
            "Isotropic Resampling...", callback=self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " isotropic"
        self.imageLayer = self.widget.getImageLayer("image")
        scale = [self.imageLayer.scale[1]] * len(self.imageLayer.scale)
        self.viewer.add_image(
            self.operation.result,
            name=name,
            scale=scale,
            units=self.imageLayer.units,
            blending="additive",
            colormap=self.imageLayer.colormap,
        )


# noinspection PyTypeChecker
class MedianFilterWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer, sameRowSet={"size z", "radius"})

    def getOptions(self):
        options = Options("Filament Toolbox", "median_filter")
        options.addImage()
        options.addInt("size xy", value=3)
        options.addInt("size z", value=1)
        options.addChoice(
            "footprint",
            choices=["none", "cube", "ball", "octahedron"],
            value="none",
        )
        options.addInt("radius", value=1)
        options.addChoice("mode", choices=self.modes, value=self.modes[0])
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = MedianFilter(self.imageLayer.data)
        self.operation.size = (
            self.options.value("size z"),
            self.options.value("size xy"),
            self.options.value("size xy"),
        )
        footprintText = self.options.value("footprint")
        radius = self.options.value("radius")
        footprint = self.getFootprintFunction(
            footprintText, footprintRadius=radius
        )
        self.operation.footprint = footprint
        self.operation.mode = self.options.value("mode")
        self.runOperationInThread(
            "Applying Median Filter...", callback=self.displayResult
        )

    @classmethod
    def getFootprintFunction(cls, footprintText, footprintRadius=1):
        footprint = None
        if not footprintText == "none":
            if footprintText == "cube":
                footprintWidth = 2 * footprintRadius + 1
                footprint = skimage.morphology.footprint_rectangle(
                    (footprintWidth, footprintWidth, footprintWidth)
                )
            else:
                footprintFunction = getattr(skimage.morphology, footprintText)
                footprint = footprintFunction(footprintRadius)
        return footprint

    def displayResult(self):
        name = self.imageLayer.name + " Median"
        self.displayImage(name)


class RollingBallWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament Toolbox", "rolling_ball")
        options.addImage()
        options.addInt("radius", value=25)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = RollingBall(self.imageLayer.data)
        self.operation.radius = self.options.value("radius")
        self.runOperationInThread(
            "Applying Rolling Ball...", callback=self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " RollingBall"
        self.displayImage(name)


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
        input_layer_label, self.input_layer_combo_box = (
            WidgetTool.getComboInput(
                self,
                "image:",
                self.image_layers,
            )
        )
        self.input_layer_combo_box.currentIndexChanged.connect(
            self.update_current_layer
        )
        self.min_value_slider = QSlider(Qt.Horizontal)
        self.min_value_slider.valueChanged.connect(self.min_threshold_changed)
        min_value_label, self.min_value_input = WidgetTool.getLineInput(
            self,
            "min.:",
            self.min_value,
            self.field_width,
            self.min_value_input_changed,
        )
        self.min_value_input.textChanged.connect(self.min_value_input_changed)
        self.max_value_slider = QSlider(Qt.Horizontal)
        self.max_value_slider.valueChanged.connect(self.max_threshold_changed)
        max_value_label, self.max_value_input = WidgetTool.getLineInput(
            self,
            "max.:",
            self.max_value,
            self.field_width,
            self.max_value_input_changed,
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
        new_layer = self.napari_util.getLayerWithName(
            self.input_layer_combo_box.currentText()
        )
        if (
            not new_layer
            or not isinstance(new_layer, Image)
            or new_layer is self.current_layer
        ):
            return
        if self.current_layer:
            self.current_layer.colormap = self.original_cmap
            self.current_layer.blending = self.original_blending
        self.current_layer = new_layer
        self.original_cmap = new_layer.colormap
        self.original_blending = new_layer.blending
        new_layer.colormap = "HiLo"
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
        if "float" in str(new_layer.data.dtype):
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
        if (
            value >= max_threshold_value
            and max_threshold_value < max_threshold_limit
        ):
            self.max_value_slider.setValue(value + 1)
        if value >= max_threshold_value == max_threshold_limit:
            new_value = value - 1
        text = self.input_layer_combo_box.currentText()
        layer = self.napari_util.getLayerWithName(text)
        data_min = np.min(layer.data)
        data_max = np.max(layer.data)
        new_value_float = new_value
        if "float" in str(layer.data.dtype):
            new_value_float = data_min + (
                (new_value / 65535.0) * (data_max - data_min)
            )
            max_threshold_value = data_min + (
                (max_threshold_value / 65535.0) * (data_max - data_min)
            )
        self.current_layer.contrast_limits = [
            new_value_float,
            max_threshold_value,
        ]
        self.min_value_input.setText(str(new_value))

    def max_threshold_changed(self, value):
        new_value = value
        min_threshold_value = self.min_value_slider.value()
        min_threshold_limit = self.max_value_slider.minimum()
        if (
            value <= min_threshold_value
            and min_threshold_value > min_threshold_limit
        ):
            self.min_value_slider.setValue(value - 1)
        if value <= min_threshold_value == min_threshold_limit:
            new_value = value + 1
        text = self.input_layer_combo_box.currentText()
        layer = self.napari_util.getLayerWithName(text)
        data_min = np.min(layer.data)
        data_max = np.max(layer.data)
        new_value_float = new_value
        if "float" in str(layer.data.dtype):
            new_value_float = data_min + (
                (new_value / 65535.0) * (data_max - data_min)
            )
            min_threshold_value = data_min + (
                (min_threshold_value / 65535.0) * (data_max - data_min)
            )
        self.current_layer.contrast_limits = [
            min_threshold_value,
            new_value_float,
        ]
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
        if "float" in str(self.input_layer.data.dtype):
            min_value = data_min + (
                (min_value / 65535.0) * (data_max - data_min)
            )
            max_value = data_min + (
                (max_value / 65535.0) * (data_max - data_min)
            )
        self.filter = Threshold(self.current_layer.data)
        self.filter.min_value = min_value
        self.filter.max_value = max_value
        worker = create_worker(
            self.filter.run, _progress={"desc": "Thresholding image..."}
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
            blending="additive",
        )


class RidgeFilterWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer", sameRowSet=None):
        self.sigmas = [1, 3]
        super().__init__(viewer, sameRowSet)

    def addSigmaOption(self, options):
        options.addStr(
            "sigmas", value=self.getSigmasAsText(), callback=self.sigmasChanged
        )

    def sigmasChanged(self, value):
        self.sigmas = value.strip().split(",")
        self.sigmas = [float(sigma.strip()) for sigma in self.sigmas]

    @classmethod
    def addBlackRidgesOption(self, options):
        options.addBool("black ridges", False)

    def getSigmasAsText(self):
        return ",".join([str(sigma) for sigma in self.sigmas])

    def apply(self):
        raise Exception(
            "Abstract method apply of class RidgeFilterWidget called!"
        )

    def displayResult(self):
        raise Exception(
            "Abstract method displayResult of class RidgeFilterWidget called!"
        )

    def getOptions(self):
        raise Exception(
            "Abstract method getOptions of class RidgeFilterWidget called!"
        )


class FrangiFilterWidget(RidgeFilterWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="frangi_filter"
        )
        options.addImage()
        self.addSigmaOption(options)
        options.addFloat("alpha", value=0.5)
        options.addFloat("beta", value=0.5)
        options.addStr("gamma", value="None")
        self.addBlackRidgesOption(options)
        self.addModesOption(options)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = FrangiFilter(self.imageLayer.data)
        self.operation.sigmas = self.sigmas
        self.operation.alpha = self.options.value("alpha")
        self.operation.beta = self.options.value("beta")
        gamma = None
        gammaText = self.options.value("gamma").strip().lower()
        if not gammaText == "none":
            gamma = float(gammaText)
        self.operation.gamma = gamma
        self.operation.black_ridges = self.options.value("black ridges")
        self.operation.mode = self.options.value("mode")
        self.runOperationInThread(
            "Applying Frangi Filter...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " Frangi"
        self.displayImage(name)


class SatoFilterWidget(RidgeFilterWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="sato_filter"
        )
        options.addImage()
        self.addSigmaOption(options)
        self.addBlackRidgesOption(options)
        self.addModesOption(options)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = SatoFilter(self.imageLayer.data)
        self.operation.sigmas = self.sigmas
        self.operation.black_ridges = self.options.value("black ridges")
        self.operation.mode = self.options.value("mode")
        self.runOperationInThread(
            "Applying Sato Filter...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " Sato"
        self.displayImage(name)


class MeijeringFilterWidget(RidgeFilterWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="meijering_filter"
        )
        options.addImage()
        self.addSigmaOption(options)
        options.addStr("alpha", value="None")
        self.addBlackRidgesOption(options)
        self.addModesOption(options)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = MeijeringFilter(self.imageLayer.data)
        self.operation.sigmas = self.sigmas
        alpha = None
        alphaText = self.options.value("alpha").strip().lower()
        if not alphaText == "none":
            alpha = float(alphaText)
        self.operation.alpha = alpha
        self.operation.black_ridges = self.options.value("black ridges")
        self.operation.mode = self.options.value("mode")
        self.runOperationInThread(
            "Applying Meijering Filter...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " Meijering"
        self.displayImage(name)


class DilationWidget(MorphologySimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer, sameRowSet={"radius"})

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="dilate"
        )
        options.addImage()
        self.addFootprintOptions(options)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = Dilation(self.imageLayer.data)
        self.operation.mode = self.options.value("mode")
        footprint = self.getFootprint(
            self.options.value("footprint"),
            self.options.value("radius"),
            self.imageLayer.data.ndim,
        )
        self.operation.footprint = footprint
        self.runOperationInThread("Applying Dilation...", self.displayResult)

    def displayResult(self):
        name = self.imageLayer.name + " dilation"
        self.displayImage(name)


class ClosingWidget(MorphologySimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer, sameRowSet={"radius"})

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="close"
        )
        options.addImage()
        self.addFootprintOptions(options)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = Closing(self.imageLayer.data)
        self.operation.mode = self.options.value("mode")
        footprint = self.getFootprint(
            self.options.value("footprint"),
            self.options.value("radius"),
            self.imageLayer.data.ndim,
        )
        self.operation.footprint = footprint
        self.runOperationInThread("Applying Closing...", self.displayResult)

    def displayResult(self):
        name = self.imageLayer.name + " close"
        self.displayImage(name)


class LabelWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.widget.widgets["image"][1].currentTextChanged.connect(
            self.onSelectedImageChanged
        )

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="label"
        )
        options.addImage()
        options.addChoice(
            "connectivity",
            choices=["1", "2", "3"],
        )
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = Label(self.imageLayer.data)
        connectivity = int(self.options.value("connectivity"))
        if self.imageLayer.data.ndim == 2 and connectivity == 3:
            connectivity = 2
        self.operation.connectivity = connectivity
        self.runOperationInThread("Labeling...", self.displayResult)

    def displayResult(self):
        name = self.imageLayer.name + " labels"
        self.displayLabels(name)

    def onSelectedImageChanged(self, text):
        napariUtil = NapariUtil(self.viewer)
        layer = napariUtil.getLayerWithName(text)
        if not layer:
            return
        if not isinstance(layer, Image):
            return
        self.widget.widgets["connectivity"][1].setCurrentText(
            str(layer.data.ndim)
        )


class RemoveSmallObjectsWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options(
            "Filament Toolbox", optionsName="remove_small_objects"
        )
        options.addLabels()
        options.addInt("max. size", value=64)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("labels")
        self.operation = RemoveSmallObjects(self.imageLayer.data)
        self.operation.min_size = self.options.value("max. size")
        self.runOperationInThread(
            "Removing small objects...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " small objects removed"
        self.displayLabels(name)


class ClearBorderWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament toolbox", "clear border")
        options.addLabels()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("labels")
        self.operation = ClearBorder(self.imageLayer.data)
        self.runOperationInThread(
            "Running Clear Border...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " cleared border"
        self.displayLabels(name)


class SkeletonizeWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.widget.widgets["labels"][1].currentTextChanged.connect(
            self.onLayerChanged
        )

    def getOptions(self):
        options = Options("Filament toolbox", "skeletonize")
        options.addLabels()
        options.addChoice("method", choices=["lee", "zhang"], value="lee")
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("labels")
        self.operation = Skeletonize(self.imageLayer.data)
        self.operation.method = self.options.value("method")
        self.runOperationInThread("Skeletonizing...", self.displayResult)

    def displayResult(self):
        name = (
            self.imageLayer.name + " skeleton-" + self.options.value("method")
        )
        self.displayImage(name)

    def onLayerChanged(self):
        layer = self.widget.getImageLayer("labels")
        if not isinstance(layer, Labels):
            return
        if (
            layer.data.ndim == 3
            and self.widget.widgets["method"][1].currentText() == "zhang"
        ):
            self.widget.widgets["method"][1].setCurrentText("lee")


class HamiltonJacobiSkeletonizeWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament Toolbox", "hamilton_jacobi_skeletonize")
        options.addLabels()
        options.addFloat("flux threshold", value=2.5)
        options.addFloat("dilation", value=1.5)
        options.addBool("use anisotropic diffusion", value=False)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("labels")
        self.operation = HamiltonJacobiSkeleton(self.imageLayer.data)
        self.operation.flux_threshold = self.options.value("flux threshold")
        self.operation.dilation = self.options.value("dilation")
        self.operation.use_anisotropic_diffusion = self.options.value(
            "use_anisotropic_diffusion"
        )
        self.runOperationInThread(
            "Calculating Hamilton Jacobi Skeleton...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " HJS"
        self.displayLabels(name)


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
        self.estimators = 50
        self.max_depth = 10
        self.sigma_min_input = None
        self.sigma_max_input = None
        self.num_sigma_input = None
        self.pixelClassifier = None
        self.estimators_input = None
        self.max_depth_input = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)
        self.point_combo_boxes.append(self.point_layer_combo_box)

    def create_layout(self):
        main_layout = QVBoxLayout()
        input_layer_label, self.input_layer_combo_box = (
            WidgetTool.getComboInput(
                self,
                "image:",
                self.image_layers,
            )
        )
        point_layer_label, self.point_layer_combo_box = (
            WidgetTool.getComboInput(
                self,
                "points:",
                self.point_layers,
            )
        )
        self.intensity_check_box = QCheckBox("Intensity features")
        self.intensity_check_box.setChecked(self.intensity_features)
        self.edges_check_box = QCheckBox("Edges Features")
        self.edges_check_box.setChecked(self.edges_features)
        self.texture_check_box = QCheckBox("Texture Features")
        self.texture_check_box.setChecked(self.texture_features)
        sigma_min_label, self.sigma_min_input = WidgetTool.getLineInput(
            self,
            "sigma min.:",
            self.sigma_min,
            self.field_width,
            self.sigma_changed,
        )
        sigma_max_label, self.sigma_max_input = WidgetTool.getLineInput(
            self,
            "sigma max.:",
            self.sigma_max,
            self.field_width,
            self.sigma_changed,
        )
        num_sigma_label, self.num_sigma_input = WidgetTool.getLineInput(
            self,
            "num. sigma:",
            self.num_sigma,
            self.field_width,
            self.sigma_changed,
        )
        train_button = QPushButton("&Train")
        train_button.clicked.connect(self.on_train_button_clicked)
        classify_button = QPushButton("&Classify")
        classify_button.clicked.connect(self.on_classify_button_clicked)

        estimators_label, self.estimators_input = WidgetTool.getLineInput(
            self,
            "estimators:",
            self.estimators,
            self.field_width,
            self.estimators_changed,
        )
        max_depth_label, self.max_depth_input = WidgetTool.getLineInput(
            self,
            "max. depth:",
            self.max_depth,
            self.field_width,
            self.max_depth_changed,
        )
        layer_layout = QHBoxLayout()
        point_layout = QHBoxLayout()
        checkboxes_layout = QVBoxLayout()
        sigma_min_layout = QHBoxLayout()
        sigma_max_layout = QHBoxLayout()
        num_sigma_layout = QHBoxLayout()
        button_layout = QVBoxLayout()
        estimators_layout = QHBoxLayout()
        max_depth_layout = QHBoxLayout()

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
        estimators_layout.addWidget(estimators_label)
        estimators_layout.addWidget(self.estimators_input)
        max_depth_layout.addWidget(max_depth_label)
        max_depth_layout.addWidget(self.max_depth_input)

        main_layout.addLayout(layer_layout)
        main_layout.addLayout(point_layout)
        main_layout.addLayout(checkboxes_layout)
        main_layout.addLayout(sigma_min_layout)
        main_layout.addLayout(sigma_max_layout)
        main_layout.addLayout(num_sigma_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(estimators_layout)
        main_layout.addLayout(max_depth_layout)

        self.setLayout(main_layout)

    def sigma_changed(self):
        pass

    def estimators_changed(self):
        pass

    def max_depth_changed(self):
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
        if num_sigma in ["None", "NONE", "none"]:
            num_sigma = None
        else:
            num_sigma = int(num_sigma)
        estimators = int(self.estimators_input.text().strip())
        max_depth = int(self.max_depth_input.text().strip())

        self.pixelClassifier = RandomForestPixelClassifier(
            self.input_layer.data
        )
        self.pixelClassifier.training_points = point_layer.data
        self.pixelClassifier.training_points_classes = [
            str(e) for e in point_layer.face_color
        ]
        self.pixelClassifier.intensity = use_intensity
        self.pixelClassifier.edges = use_edges
        self.pixelClassifier.texture = use_texture
        self.pixelClassifier.sigma_min = sigma_min
        self.pixelClassifier.sigma_max = sigma_max
        self.pixelClassifier.num_sigma = num_sigma
        self.pixelClassifier.n_estimators = estimators
        self.pixelClassifier.max_depth = max_depth
        worker = create_worker(
            self.pixelClassifier.train,
            _progress={"desc": "Training Pixel Classifier..."},
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
            blending="additive",
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
            blending="additive",
        )


class BrightestPathTracingWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox",
            optionsName="Brightest Path Tracing",
        )
        options.addImage()
        options.addPoints()
        options.addChoice(
            "method", choices=["A-star", "NBA-star"], value="NBA-star"
        )
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        points = self.widget.getImageLayer("points")
        self.operation = BrightestPathTracing(
            self.imageLayer.data, points.data
        )
        self.operation.method = self.options.value("method")
        self.runOperationInThread(
            "Tracing Brightest Path...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " traces"
        self.displayLabels(name)


class MetricsWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer, sameRowSet={"clDice"})
        self.metrics = {"Dice": Dice, "clDice": CenterlineDice}
        self.table = {}
        self.results = {}
        self.layer1 = None
        self.layer2 = None

    def getOptions(self):
        options = Options("Filament Toolbox", "metrics")
        options.addLabels(name="labels 1")
        options.addLabels(name="labels 2")
        options.addBool("Dice")
        options.addBool("clDice")
        options.load()
        return options

    def apply(self):
        self.layer1 = self.widget.getImageLayer("labels 1")
        self.layer2 = self.widget.getImageLayer("labels 2")
        worker = create_worker(
            self.calculate_metrics,
            _progress={"desc": "Calculating metrics..."},
        )
        worker.finished.connect(self.displayResult)
        worker.start()

    def displayResult(self):
        if "Metrics" in self.viewer.window.dock_widgets.keys():
            self.viewer.window.remove_dock_widget(
                self.viewer.window.dock_widgets["Metrics"]
            )
            self.viewer.window.dock_widgets["Metrics"].close()
        self.viewer.window.add_dock_widget(
            TableView(self.results), name="Metrics"
        )

    def calculate_metrics(self):
        if not "image1" in self.results.keys():
            self.results["image1"] = []
        self.results["image1"].append(self.layer1.name)
        if not "image2" in self.results.keys():
            self.results["image2"] = []
        self.results["image2"].append(self.layer2.name)
        for key, value in self.metrics.items():
            if self.options.value(key):
                metric_class = self.metrics[key]
                metric = metric_class(self.layer1.data, self.layer2.data)
                metric.calculate()
                if not key in self.results.keys():
                    self.results[key] = []
                self.results[key].append(metric.result)


class EuclideanDistanceTransformWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options(
            applicationName="Filament Toolbox", optionsName="edt"
        )
        options.addImage()
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = EuclideanDistanceTransform(self.imageLayer.data)
        self.runOperationInThread(
            "Applying Euclidean Distance Transform...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " edt"
        self.displayImage(name)


class LocalThicknessWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

    def getOptions(self):
        options = Options("Filament Toolbox", "local_thickness")
        options.addImage()
        options.addFloat("scale", value=0.5)
        options.addBool("physical units (experimental)", value=False)
        options.load()
        return options

    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = LocalThickness(self.imageLayer.data)
        self.operation.scale = self.options.value("scale")
        if self.options.value("physical units (experimental)"):
            self.operation.usePhysicalUnits = self.options.value(
                "physical units (experimental)"
            )
            self.operation.spacing = self.imageLayer.scale
            self.operation.scale = 1
        self.runOperationInThread(
            "Calculating Local Thickness...", self.displayResult
        )

    def displayResult(self):
        name = self.imageLayer.name + " thickness"
        self.displayImage(name, colormap="inferno")


class MeasureLabelsWidget(SimpleWidget):

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.widget.addButton("Options", self.optionsButtonPressed)
        self.regionPropOptions = None

    def optionsButtonPressed(self):
        self.regionPropOptions = self.getRegionPropsOptions()
        self.regionPropOptions.load()
        optionsPerRow = 2
        sameRowSet = set()
        counter = 0
        allProps = self.regionPropOptions.items.keys()
        for prop in allProps:
            if not counter == 0 and not counter % optionsPerRow == 0:
                sameRowSet.add(prop)
            counter = counter + 1
        widget = OptionsWidget(
            self.viewer,
            self.regionPropOptions,
            layout_type="grid",
            client=self,
            sameRowSet=sameRowSet,
        )
        widget.addOKButton(None)
        widget.addApplyButton(None)
        widget.addCancelButton(None)
        self.viewer.window.add_dock_widget(
            widget, area="right", tabify=True, name="Options of Measure Labels"
        )

    def getOptions(self):
        options = Options("Filament Toolbox", "measure_labels")
        options.addLabels()
        options.addImage(optional=[True, False])
        options.load()
        return options

    @classmethod
    def getRegionPropsOptions(cls):
        options = Options("Filament Toolbox", "region_props")
        allProps = MeasureLabels.getAllProperties()
        for prop in allProps:
            options.addBool(prop, value=False)
        options.setValue("label", True)
        options.setValue("area", True)
        options.load()
        return options

    def apply(self):
        self.regionPropOptions = self.getRegionPropsOptions()
        self.imageLayer = self.widget.getImageLayer("labels")
        intensityImage = self.widget.getImageLayer("image")
        intensityData = None
        if intensityImage:
            intensityData = intensityImage.data
        self.operation = MeasureLabels(
            self.imageLayer.data, intensityData, self.imageLayer.scale
        )
        self.operation.selectedProperties = set(
            [
                key
                for key in self.regionPropOptions.items.keys()
                if self.regionPropOptions.value(key)
            ]
        )
        self.runOperationInThread("Measuring Labels...", self.displayResult)

    def displayResult(self):
        self.viewer.window.add_dock_widget(
            TableView(self.operation.table),
            name="Measurements of " + self.imageLayer.name,
        )
