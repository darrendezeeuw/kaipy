import pytest
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
from kaipy.gamera.deltabViz import GenUniformLL
from kaipy.gamera.deltabViz import DecorateDBAxis, GetCoords, CheckLevel, GenTStr

# FILE: kaipy/gamera/test_deltabViz.py

class MockDbData:
	def __init__(self, X, Y, Z):
		self.X = X
		self.Y = Y
		self.Z = Z

def test_GenUniformLL_basic():
	X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Z = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	dbdata = MockDbData(X, Y, Z)
	
	LatI, LonI, LatC, LonC = GenUniformLL(dbdata, k0=0)
	assert LatI.shape == (2,)
	assert LonI.shape == (2,)
	assert LatC.shape == (1,)
	assert LonC.shape == (1,)

def test_GenUniformLL_values():
	X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Z = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	dbdata = MockDbData(X, Y, Z)
	
	LatI, LonI, LatC, LonC = GenUniformLL(dbdata, k0=0)
	np.testing.assert_almost_equal(LatI, [35.26438968, 35.26438968])
	np.testing.assert_almost_equal(LonI, [0., 360.])
	np.testing.assert_almost_equal(LatC, [35.26438968])
	np.testing.assert_almost_equal(LonC, [180.])

def test_GenUniformLL_different_k0():
	X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Z = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	dbdata = MockDbData(X, Y, Z)
	
	LatI, LonI, LatC, LonC = GenUniformLL(dbdata, k0=1)
	assert LatI.shape == (2,)
	assert LonI.shape == (2,)
	assert LatC.shape == (1,)
	assert LonC.shape == (1,)

	
