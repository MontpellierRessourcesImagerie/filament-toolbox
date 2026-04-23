import numpy as np

from filament_toolbox._widget import (
    MedianFilterWidget
)


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    image = np.random.random((100, 100))
    viewer.add_image(image, name="image")
    # create our widget, passing in the viewer
    my_widget = MedianFilterWidget(viewer)
    viewer.window.add_dock_widget(my_widget)

    # call our widget method

    options = my_widget.options
    print(options.items)
    assert True
