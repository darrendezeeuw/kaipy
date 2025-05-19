import pytest
import numpy as np
import h5py

# FILE: kaipy/gamera/test_gamGrids.py

from kaipy.gamera.gamGrids import (
	genEllip, genSph, genEgg, genFatEgg, RampUp, Egglipses,
	GenKSph, GenKSphNonU, GenKSphNonUGL, Aug2D, Aug2Dext, genRing,
	PrintRing, Aug3D, WriteGrid, WriteChimp, VizGrid, LoadTabG, regrid
)

def test_genEllip():
	XX, YY = genEllip()
	assert XX.shape == (33, 65)
	assert YY.shape == (33, 65)

def test_genSph():
	XX, YY = genSph()
	assert XX.shape == (33, 65)
	assert YY.shape == (33, 65)

def test_genEgg():
	xx, yy = genEgg()
	assert xx.shape == (33, 65)
	assert yy.shape == (33, 65)

def test_genFatEgg():
	xx, yy = genFatEgg()
	assert xx.shape == (33, 65)
	assert yy.shape == (33, 65)

def test_RampUp():
	assert RampUp(0, 1, 1) == 0
	assert RampUp(2, 1, 1) == 1
	assert 0 < RampUp(1, 0.5, 1) < 1

def test_Egglipses():
	x0, a, b = Egglipses()
	assert x0.shape == (33,)
	assert a.shape == (33,)
	assert b.shape == (33,)

def test_GenKSph():
	X3, Y3, Z3 = GenKSph()
	assert X3.shape == (41, 73, 41)
	assert Y3.shape == (41, 73, 41)
	assert Z3.shape == (41, 73, 41)

def test_GenKSphNonU():
	X3, Y3, Z3 = GenKSphNonU()
	assert X3.shape == (41, 73, 41)
	assert Y3.shape == (41, 73, 41)
	assert Z3.shape == (41, 73, 41)

def test_GenKSphNonUGL():
	X3, Y3, Z3 = GenKSphNonUGL()
	assert X3.shape == (203, 73, 41)
	assert Y3.shape == (203, 73, 41)
	assert Z3.shape == (203, 73, 41)

def test_Aug2D():
	XX = np.array([[1, 2], [3, 4]])
	YY = np.array([[5, 6], [7, 8]])
	xxG, yyG = Aug2D(XX, YY)
	assert xxG.shape == (10, 10)
	assert yyG.shape == (10, 10)

def test_Aug2Dext():
	XX = np.array([[1, 2], [3, 4]])
	YY = np.array([[5, 6], [7, 8]])
	xxG, yyG = Aug2Dext(XX, YY, 2)
	assert xxG.shape == (4, 2)
	assert yyG.shape == (4, 2)

def test_genRing():
	XX = np.array([[1, 2], [3, 4]])
	YY = np.array([[5, 6], [7, 8]])
	genRing(XX, YY)

def test_PrintRing():
	NCh = np.array([8, 16, 32])
	PrintRing(NCh)

def test_Aug3D():
	Ni = 32
	Nj = 64
	xxG = np.fromfunction(lambda i, j: i * 100 + j, (Ni, Nj))
	yyG = np.fromfunction(lambda i, j: i * 100 + j, (Ni, Nj))
	X3, Y3, Z3 = Aug3D(xxG, yyG)
	assert X3.shape == (Ni, Nj, 41)
	assert Y3.shape == (Ni, Nj, 41)
	assert Z3.shape == (Ni, Nj, 41)

def test_WriteGrid(tmpdir):
	file_path = tmpdir.join("test.h5")
	X3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Y3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Z3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	WriteGrid(X3, Y3, Z3, fOut=str(file_path))
	assert file_path.exists()
	with h5py.File(str(file_path), 'r') as f:
		assert 'X' in f
		assert 'Y' in f
		assert 'Z' in f

def test_WriteChimp(tmpdir):
	file_path = tmpdir.join("test.h5")
	X3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Y3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	Z3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	WriteChimp(X3, Y3, Z3, fOut=str(file_path))
	assert file_path.exists()
	with h5py.File(str(file_path), 'r') as f:
		assert 'X' in f
		assert 'Y' in f
		assert 'Z' in f

def test_VizGrid(tmpdir):
	file_path = tmpdir.join("grid.png")
	XX = np.array([[1, 2], [3, 4]])
	YY = np.array([[5, 6], [7, 8]])
	VizGrid(XX, YY, fOut=str(file_path))
	assert file_path.exists()

def test_LoadTabG():
	xxi, yyi = LoadTabG()
	assert xxi.shape == (213, 193)
	assert yyi.shape == (213, 193)

def test_regrid():
	Ni = 16
	Nj = 32
	xxi = np.fromfunction(lambda i, j: i * 100 + j, (Ni, Nj))
	yyi = np.fromfunction(lambda i, j: i * 100 + j, (Ni, Nj))
	XXi, YYi = regrid(xxi, yyi, 2*Ni, 2*Nj)
	assert XXi.shape == (33, 65)
	assert YYi.shape == (33, 65)