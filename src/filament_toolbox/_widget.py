"""
This module contains the tools of the filament-toolbox
"""

from typing import TYPE_CHECKING
from qtpy.QtWidgets import QPushButton, QWidget
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout
import skimage.morphology
from napari.utils.events import Event
from napari.qt.threading import create_worker
from filament_toolbox.lib.qtutil import WidgetTool
from filament_toolbox.lib.napari_util import NapariUtil
from filament_toolbox.lib.filter import MedianFilter, GaussianFilter

if TYPE_CHECKING:
    import napari



class ToolboxWidget(QWidget):


    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.field_width = 50
        self.napari_util = NapariUtil(self.viewer)
        self.imageLayers = self.napari_util.getImageLayers()
        self.image_combo_boxes = []
        self.input_layer_combo_box = None
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
        for combo_box in self.image_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, image_layers)



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
                                                                                 self.imageLayers,
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
                footprint_radius = 2 * footprint_radius + 1
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
                                                                                 self.imageLayers,
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
        self.sigma_iterations_input = None
        self.kappa_input = None
        self.gamma_input = None
        self.step_xy_input = None
        self.step_z_input = None
        self.options = ["favour high contrast edges", " favour wide regions "]
        self.options_combo_box = None
        self.create_layout()
        self.image_combo_boxes.append(self.input_layer_combo_box)


    def create_layout(self):
        main_layout = QVBoxLayout()
        """input_layer_label, self.input_layer_combo_box = WidgetTool.getComboInput(self, "image:",
                                                                                 self.imageLayers,
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
        )"""