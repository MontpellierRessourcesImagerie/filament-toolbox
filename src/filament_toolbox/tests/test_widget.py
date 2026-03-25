import numpy as np

from filament_toolbox._widget import (
    MedianFilterWidget
)


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    image = np.random.random((100, 100))
    value = image[50,50]
    viewer.add_image(image)

    # create our widget, passing in the viewer
    my_widget = MedianFilterWidget(viewer)

    # call our widget method

    my_widget.on_apply_button_clicked()

    assert image is my_widget.input_layer.data
