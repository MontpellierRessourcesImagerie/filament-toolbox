import napari

import filament_toolbox

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._widget import (
    MedianFilterWidget,
    GaussianFilterWidget,
    AnisotropicDiffusionFilterWidget,
    RollingBallWidget,
    ThresholdWidget,
    FrangiFilterWidget,
    SatoFilterWidget,
    MeijeringFilterWidget,
    DilationWidget,
    rgb_to_8bit,
    rgb_to_16bit,
    ClosingWidget,
    activate,
    LabelWidget,
    RemoveSmallObjectsWidget,
    ClearBorderWidget,
    SkeletonizeWidget,
    HamiltonJacobiSkeletonizeWidget,
    PixelClassifierWidget,
    measure_skeleton,
    BrightestPathTracingWidget,
    MetricsWidget,
    EuclideanDistanceTransformWidget,
    LocalThicknessWidget,
    MeasureLabels,
)


__all__ = (
    "napari_get_reader",
    "make_sample_data",
    "MedianFilterWidget",
    "MedianFilterWidget",
    "AnisotropicDiffusionFilterWidget",
    "RollingBallWidget",
    "ThresholdWidget",
    "FrangiFilterWidget",
    "SatoFilterWidget",
    "MeijeringFilterWidget",
    "DilationWidget",
    "rgb_to_8bit",
    "rgb_to_16bit",
    "ClosingWidget",
    "activate",
    "LabelWidget",
    "RemoveSmallObjectsWidget",
    "ClearBorderWidget",
    "SkeletonizeWidget",
    "HamiltonJacobiSkeletonizeWidget",
    "PixelClassifierWidget",
    "measure_skeleton",
    "BrightestPathTracingWidget",
    "MetricsWidget",
    "EuclideanDistanceTransformWidget",
    "LocalThicknessWidget",
    "MeasureLabels",
)


@napari.Viewer.bind_key('t')
def toggle_widget(param):
    print(param)
    viewer = napari.current_viewer()
    viewer.window.add_dock_widget(ThresholdWidget(viewer))
