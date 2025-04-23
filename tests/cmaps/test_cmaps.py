import pytest
import numpy as np
import os
import matplotlib as mpl
import kaipy.cmaps.kaimaps as kaimaps
from kaipy.cmaps.kaimaps import load_colormap_from_file

@pytest.fixture
def colormap_file(tmpdir):
    file_path = tmpdir.join("test_colormap.txt")
    with open(file_path, 'w') as f:
        f.write("R G B\n")
        f.write("255 0 0\n")
        f.write("0 255 0\n")
        f.write("0 0 255\n")
    return str(file_path)

def test_load_colormap_from_file(colormap_file):
    cmap = load_colormap_from_file(colormap_file)
    assert isinstance(cmap, mpl.colors.ListedColormap)
    expected_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(cmap.colors, expected_colors)

def test_load_colormap_from_file_invalid_path():
    with pytest.raises(OSError):
        load_colormap_from_file("invalid_path.txt")

def test_cmDiv_colormap():
    fIn = os.path.join(os.path.dirname(kaimaps.__file__), "cmDDiv.txt")
    if os.path.exists(fIn):
        cmDiv = load_colormap_from_file(fIn)
        assert isinstance(cmDiv, mpl.colors.ListedColormap)
    else:
        pytest.skip("cmDDiv.txt file not found")

def test_cmMLT_colormap():
    fIn = os.path.join(os.path.dirname(kaimaps.__file__), "cmMLT.txt")
    if os.path.exists(fIn):
        cmMLT = load_colormap_from_file(fIn)
        assert isinstance(cmMLT, mpl.colors.ListedColormap)
    else:
        pytest.skip("cmMLT.txt file not found")