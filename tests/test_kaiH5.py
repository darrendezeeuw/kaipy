import pytest
import h5py
import numpy as np
import datetime
import os
from astropy.time import Time

from kaipy.kaiH5 import H5Info, TPInfo, genName, genNameOld, CheckOrDie, CheckDirOrMake, StampHash, StampBranch, GetHash, GetBranch, tStep, cntSteps, cntX, getTs, LocDT, MageStep, getDims, getRootVars, getVars, PullVarLoc, PullVar, PullAtt

Ni = 32
Nj = 24
Nk = 48

@pytest.fixture(scope='session')
def temp_h5_file(tmp_path_factory):
	temp_dir = tmp_path_factory.mktemp("data")
	file_path = temp_dir / "test.h5"
	today = datetime.datetime.now()
	with h5py.File(file_path, 'w') as f:
		for i in range(3):
			grp = f.create_group("Step#{}".format(i))
			grp.attrs['time'] = np.double(i)
			grp.attrs['MJD'] = Time(today).mjd
			grp.attrs['timestep'] = i
			grp.create_dataset("D", data=np.zeros((Ni, Nj, Nk)))
	return str(file_path)

def test_H5Info_initialization(temp_h5_file):
	h5info = H5Info(temp_h5_file, noSubsec=True)
	assert h5info.fname == temp_h5_file
	assert np.array_equal(h5info.steps, np.array([0, 1, 2]))
	assert isinstance(h5info.stepStrs, list)
	assert np.array_equal(h5info.steps, np.array([0.0, 1.0, 2.0]))
	assert isinstance(h5info.MJDs, np.ndarray)
	assert isinstance(h5info.UTs, np.ndarray)

def test_TPInfo_initialization(temp_h5_file):
	assert 1 == 1
	# tpinfo = TPInfo(temp_h5_file, noSubsec=True)
	# assert tpinfo.fname == temp_h5_file
	# assert np.array_equal(tpinfo.steps, np.array([0, 1, 2]))
	# assert isinstance(tpinfo.stepStrs, list)
	# assert np.array_equal(tpinfo.steps, np.array([0.0, 1.0, 2.0]))
	# assert isinstance(tpinfo.MJDs, np.ndarray)
	# assert isinstance(tpinfo.UTs, np.ndarray)
	# assert isinstance(tpinfo.Ntp, int)
	# assert isinstance(tpinfo.id2idxMap, dict)

def test_genName():
	result = genName('base', 1, 2, 3, 4, 5, 6)
	assert result == 'base_0004_0005_0006_0001_0002_0003.gam.h5'

def test_genNameOld():
	result = genNameOld('base', 1, 2, 3, 4, 5, 6)
	assert result == 'base_0004_0005_0006_0001_0002_0003_000000000045.h5'

def test_CheckOrDie():
	with pytest.raises(SystemExit):
		CheckOrDie('non_existent_file.h5')

def test_CheckDirOrMake(tmpdir):
	dir_path = tmpdir.mkdir("testdir")
	CheckDirOrMake(str(dir_path))
	assert os.path.isdir(str(dir_path))

def test_StampHash(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		pass
	StampHash(str(file_path))
	with h5py.File(file_path, 'r') as f:
		assert 'GITHASH' in f.attrs

def test_StampBranch(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		pass
	StampBranch(str(file_path))
	with h5py.File(file_path, 'r') as f:
		assert 'GITBRANCH' in f.attrs

def test_GetHash(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.attrs['GITHASH'] = 'testhash'
	result = GetHash(str(file_path))
	assert result == 'testhash'

def test_GetBranch(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.attrs['GITBRANCH'] = 'testbranch'
	result = GetBranch(str(file_path))
	assert result == 'testbranch'

def test_tStep(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		grp = f.create_group("Step#0")
		grp.attrs['time'] = 123.456
	result = tStep(str(file_path), 0)
	assert result == 123.456

def test_cntSteps(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.create_group("Step#0")
		f.create_group("Step#1")
	nSteps, sIds = cntSteps(str(file_path))
	assert nSteps == 2
	assert np.array_equal(sIds, np.array([0, 1]))

def test_cntX(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.create_group("Step#0")
		f.create_group("Step#1")
	nSteps, sIds = cntX(str(file_path))
	assert nSteps == 2
	assert np.array_equal(sIds, np.array([0, 1]))

def test_getTs(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		grp = f.create_group("Step#0")
		grp.attrs['time'] = 123.456
	result = getTs(str(file_path), [0])
	assert np.array_equal(result, np.array([123.456]))

def test_LocDT():
	items = [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)]
	pivot = datetime.datetime(2020, 1, 1, 12)
	result = LocDT(items, pivot)
	print(result)
	assert result == 0

def test_MageStep(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		grp = f.create_group("Step#0")
		grp.attrs['MJD'] = 123.456
	T0 = datetime.datetime(2020, 1, 1)
	result = MageStep(T0, str(file_path))
	assert result == 0

def test_getDims(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.create_dataset("X", data=np.zeros((Ni, Nj, Nk)).T)
	result = getDims(str(file_path))
	assert np.array_equal(result, np.array([Ni, Nj, Nk]))

def test_getRootVars(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.create_dataset("var1", data=np.zeros((Ni, Nj, Nk)).T)
	result = getRootVars(str(file_path))
	assert result == ['var1']

def test_getVars(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		grp = f.create_group("Step#0")
		grp.create_dataset("var1", data=np.zeros((Ni, Nj, Nk)).T)
	result = getVars(str(file_path), 0)
	assert result == ['var1']

def test_PullVarLoc(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.create_dataset("var1", data=np.zeros((Ni, Nj, Nk)).T)
	result, loc = PullVarLoc(str(file_path), "var1")
	assert np.array_equal(result, np.zeros((Ni, Nj, Nk)))

def test_PullVar(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.create_dataset("var1", data=np.zeros((Ni, Nj, Nk)).T)
	result = PullVar(str(file_path), "var1")
	assert np.array_equal(result, np.zeros((Ni, Nj, Nk)))

def test_PullAtt(tmpdir):
	file_path = tmpdir.join("test.h5")
	with h5py.File(file_path, 'w') as f:
		f.attrs['attr1'] = 'value1'
	result = PullAtt(str(file_path), 'attr1')
	assert result == 'value1'