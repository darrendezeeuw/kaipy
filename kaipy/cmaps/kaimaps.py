# Standard modules
import os
import importlib.resources as pkg_resources

# Third-party modules
import matplotlib as mpl
import numpy as np

# Kaipy modules
from kaipy import cmaps

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def load_colormap_from_file(file_path):
    """
    Load a colormap from a text file.

    Args:
        file_path (str): The path to the text file containing the colormap data.

    Returns:
        matplotlib.colors.ListedColormap: The loaded colormap.
    """
    Q = np.loadtxt(file_path, skiprows=1)
    return mpl.colors.ListedColormap(Q/255.0)

# Load cmDiv colormap
fIn = pkg_resources.files(cmaps).joinpath("cmDDiv.txt")
cmDiv = load_colormap_from_file(fIn)

# Load cmMLT colormap
fIn = pkg_resources.files(cmaps).joinpath("cmMLT.txt")
cmMLT = load_colormap_from_file(fIn)


