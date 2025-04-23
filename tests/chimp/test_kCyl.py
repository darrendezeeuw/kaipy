import pytest
import numpy as np
import h5py
from kaipy.chimp.kCyl import getGrid, getSlc, PIso, getEQGrid

@pytest.fixture
def h5file(tmpdir):
	# Create a temporary HDF5 file for testing
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as hf:
		hf.create_dataset("X", data=np.random.rand(10, 10, 10))
		hf.create_dataset("Y", data=np.random.rand(10, 10, 10))
		hf.create_dataset("Z", data=np.random.rand(10, 10, 10))
		hf.create_dataset("A", data=np.random.rand(10))
		step_grp = hf.create_group("Step#0")
		step_grp.create_dataset("jPSD", data=np.random.rand(10, 10))
		step_grp.create_dataset("Pxy", data=np.random.rand(10, 10))
		step_grp.create_dataset("Pz", data=np.random.rand(10, 10))
	return str(file_path)

def test_getGrid(h5file):
	xx, yy, Ki, Kc = getGrid(h5file)
	assert xx.shape == (10, 10)
	assert yy.shape == (10, 10)
	assert Ki.shape == (10,)
	assert Kc.shape == (9,)

	xx, yy, Ki, Kc, Ai, Ac = getGrid(h5file, do4D=True)
	assert xx.shape == (10, 10)
	assert yy.shape == (10, 10)
	assert Ki.shape == (10,)
	assert Kc.shape == (9,)
	assert Ai.shape == (10,)
	assert Ac.shape == (9,)

def test_getSlc(h5file):
	V = getSlc(h5file, nStp=0, vID="jPSD")
	assert V.shape == (10, 10)

	V_wrap = getSlc(h5file, nStp=0, vID="jPSD", doWrap=True)
	assert V_wrap.shape == (10, 11)

def test_PIso(h5file):
	pR = PIso(h5file, nStp=0, pCut=1.0e-3, doAsym=False)
	assert pR.shape == (10, 10)

	pR_asym = PIso(h5file, nStp=0, pCut=1.0e-3, doAsym=True)
	assert pR_asym.shape == (10, 10)

@pytest.fixture
def h5eqfile(tmpdir):
	# Create a temporary HDF5 file for testing
	file_path = tmpdir.join("eqtest.h5")
	with h5py.File(file_path, 'w') as hf:
		hf.create_dataset("X", data=np.random.rand(10, 10))
		hf.create_dataset("Y", data=np.random.rand(10, 10))
		step_grp = hf.create_group("Step#0")
		step_grp.create_dataset("jPSD", data=np.random.rand(10, 10))
		step_grp.create_dataset("Pxy", data=np.random.rand(10, 10))
		step_grp.create_dataset("Pz", data=np.random.rand(10, 10))
	return str(file_path)

def test_getEQGrid(h5eqfile):
	xx, yy = getEQGrid(h5eqfile, doCenter=False, doWrap=False)
	assert xx.shape == (10, 10)
	assert yy.shape == (10, 10)

	xxc, yyc = getEQGrid(h5eqfile, doCenter=True, doWrap=False)
	assert xxc.shape == (9, 9)
	assert yyc.shape == (9, 9)

	xxc_wrap, yyc_wrap = getEQGrid(h5eqfile, doCenter=True, doWrap=True)
	assert xxc_wrap.shape == (9, 10)
	assert yyc_wrap.shape == (9, 10)