import pytest
import numpy as np
import h5py
import datetime
from astropy.time import Time
from kaipy.gamera.magsphere import GamsphPipe

Ni = 32
Nj = 24
Nk = 48

Nlat = 44
Nlon = 360

@pytest.fixture
def gamera_pipe(tmpdir):
    # Create a temporary directory and file structure for testing
    fdir = tmpdir.mkdir("data")
    ftag = "test"
    file_path = fdir.join("test.gam.h5")
    mjds = []
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("X", data=np.zeros((Ni+1, Nj+1, Nk+1)))
        f.create_dataset("Y", data=np.zeros((Ni+1, Nj+1, Nk+1)))
        f.create_dataset("Z", data=np.zeros((Ni+1, Nj+1, Nk+1)))
        for i in range(3):
            grp = f.create_group("Step#{}".format(i))
            grp.attrs['time'] = np.double(i)
            mjd = Time(datetime.datetime.now()).mjd
            grp.attrs['MJD'] = mjd
            mjds.append(mjd)
            grp.attrs['timestep'] = i
            grp.create_dataset("D", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("P", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("Bx", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("By", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("Bz", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("Vx", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("Vy", data=np.zeros((Ni, Nj, Nk)))
            grp.create_dataset("Vz", data=np.zeros((Ni, Nj, Nk)))

    file_path = fdir.join("test.mix.h5")
    lat = np.arange(Nlat+1)
    long = np.arange(Nlon+1)
    X = np.outer(np.sin(np.radians(lat)), np.cos(np.radians(long)))
    Y = np.outer(np.sin(np.radians(lat)), np.sin(np.radians(long)))
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)

        for i in range(3):
            grp = f.create_group("Step#{}".format(i))
            grp.attrs['time'] = np.double(i)
            grp.attrs['MJD'] = mjds[i]
            grp.attrs['timestep'] = i
            grp.create_dataset("Potential NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Potential SOUTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Field-aligned current NORTH", data=np.zeros((Nlat, Nlon)))
            grp.create_dataset("Field-aligned current SOUTH", data=np.zeros((Nlat, Nlon)))

    return GamsphPipe(str(fdir), ftag, doFast=True, uID="Earth")