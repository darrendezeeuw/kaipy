import pytest
import numpy as np
from kaipy.remix import remix
from tests.conftest import Nlat, Nlon

def test_remix_initialization(mix_file):
    r = remix.remix(mix_file, 0)
    assert r.ion is not None
    assert r.Initialized is False

def test_get_data(mix_file):
    r = remix.remix(mix_file, 0)
    data = r.get_data(mix_file, 0)
    assert 'X' in data
    assert 'Y' in data
    assert 'R' in data
    assert 'THETA' in data

def test_init_vars(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    assert r.Initialized is True
    assert 'potential' in r.variables
    assert 'current' in r.variables

def test_get_spherical(mix_file):
    r = remix.remix(mix_file, 0)
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    r, theta = r.get_spherical(x, y)
    assert r.shape == x.shape
    assert theta.shape == y.shape

def test_distance(mix_file):
    r = remix.remix(mix_file, 0)
    p0 = (0, 0, 0)
    p1 = (1, 1, 1)
    dist = r.distance(p0, p1)
    assert dist == pytest.approx(np.sqrt(3))

def test_calcFaceAreas(mix_file):
    r = remix.remix(mix_file, 0)
    x = r.ion['X']
    y = r.ion['Y']
    areas = r.calcFaceAreas(x, y)
    assert areas.shape == (Nlat, Nlon)
    assert areas[0, 0] == pytest.approx(2.658e-6,rel=1e-3)

def test_efieldmix_file(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    etheta, ephi = r.efield()
    assert etheta.shape == r.variables['potential']['data'].shape
    assert ephi.shape == r.variables['potential']['data'].shape

def test_joule(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    joule = r.joule()
    assert joule.shape == r.variables['sigmap']['data'].shape

def test_cartesianCellCenters(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    xc, yc, theta, phi = r.cartesianCellCenters()
    assert xc.shape == r.variables['potential']['data'].shape
    assert yc.shape == r.variables['potential']['data'].shape
    assert theta.shape == r.variables['potential']['data'].shape
    assert phi.shape == r.variables['potential']['data'].shape

def test_hCurrents(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    xc, yc, theta, phi, dtheta, dphi, Jh_theta, Jh_phi, Jp_theta, Jp_phi, cosDipAngle = r.hCurrents()
    assert Jh_theta.shape == r.variables['potential']['data'].shape
    assert Jh_phi.shape == r.variables['potential']['data'].shape

def test_dB(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    xyz = np.array([[1, 1, 1], [2, 2, 2]])
    dBr, dBtheta, dBphi = r.dB(xyz)
    assert dBr.shape == (1,1,2)
    assert dBtheta.shape == (1,1,2)
    assert dBphi.shape == (1,1,2)

def test_BSFluxTubeInt(mix_file):
    r = remix.remix(mix_file, 0)
    r.init_vars('north')
    xyz = np.array([[1, 1, 1], [2, 2, 2]])
    intx, inty, intz = r.BSFluxTubeInt(xyz, Rinner=2.0)
    assert intx.shape == (Nlat, Nlon, 2)
    assert inty.shape == (Nlat, Nlon, 2)
    assert intz.shape == (Nlat, Nlon, 2)