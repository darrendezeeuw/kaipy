import pytest
import numpy as np
import h5py
from kaipy.chimp.chimph5p import cntTPs, bndTPs, locPID, getH5pid, getH5p, getH5pT

@pytest.fixture
def h5file(tmpdir):
	# Create a temporary H5Part file for testing
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as hf:
		grp = hf.create_group("Step#0")
		grp.create_dataset("id", data=np.arange(1, 101))
		grp.attrs["time"] = 0.0
		for i in range(1, 10):
			grp = hf.create_group(f"Step#{i}")
			grp.create_dataset("id", data=np.arange(1, 101))
			grp.attrs["time"] = i * 10.0
	return str(file_path)

def test_cntTPs(h5file):
	assert cntTPs(h5file) == 100

def test_bndTPs(h5file):
	Np, nS, nE = bndTPs(h5file)
	assert Np == 100
	assert nS == 1
	assert nE == 100

def test_locPID(h5file):
	assert locPID(h5file, 50) == 49
	#assert locPID(h5file, 150) is None

def test_getH5pid(h5file):
	t, V = getH5pid(h5file, "id", 50)
	assert len(t) == 10
	assert len(V) == 10
	assert V[0] == 50

def test_getH5p(h5file):
	t, V = getH5p(h5file, "id")
	assert len(t) == 10
	assert V.shape == (10, 100)
	mask = np.array([True] * 50 + [False] * 50)
	t, V_masked = getH5p(h5file, "id", Mask=mask)
	assert V_masked.shape == (10, 50)

def test_getH5pT(h5file):
	V = getH5pT(h5file, "id", 0)
	assert len(V) == 100
	assert V[0] == 1
	assert V[-1] == 100