import pytest
import numpy as np
import datetime
from kaipy.transform import SMtoGSM, GSMtoSM, GSEtoGSM

def test_SMtoGSM():
    x, y, z = 1, 2, 3
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_gsm, y_gsm, z_gsm = SMtoGSM(x, y, z, ut)
    assert np.isclose(x_gsm, -0.126, atol=1e-3)
    assert np.isclose(y_gsm, 2.0, atol=1e-3)
    assert np.isclose(z_gsm, 3.159, atol=1e-3)

def test_GSMtoSM():
    x, y, z = 1, 2, 3
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_sm, y_sm, z_sm = GSMtoSM(x, y, z, ut)
    assert np.isclose(x_sm, 1.997, atol=1e-3)
    assert np.isclose(y_sm, 2.0, atol=1e-3)
    assert np.isclose(z_sm, 2.451, atol=1e-3)

def test_GSEtoGSM():
    x, y, z = 1, 2, 3
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_gsm, y_gsm, z_gsm = GSEtoGSM(x, y, z, ut)
    assert np.isclose(x_gsm, 1.0, atol=1e-3)
    assert np.isclose(y_gsm, 0.540, atol=1e-3)
    assert np.isclose(z_gsm, 3.565, atol=1e-3)

def test_GSEtoGSM():
    x, y, z = 1, 2, 3
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_gsm, y_gsm, z_gsm = GSEtoGSM(x, y, z, ut)
    assert np.isclose(x_gsm, 1.0, atol=1e-3)
    assert np.isclose(y_gsm, 0.540, atol=1e-3)
    assert np.isclose(z_gsm, 3.565, atol=1e-3)

def test_GSEtoGSM_zero():
    x, y, z = 0, 0, 0
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_gsm, y_gsm, z_gsm = GSEtoGSM(x, y, z, ut)
    assert np.isclose(x_gsm, 0.0, atol=1e-3)
    assert np.isclose(y_gsm, 0.0, atol=1e-3)
    assert np.isclose(z_gsm, 0.0, atol=1e-3)

def test_GSEtoGSM_negative():
    x, y, z = -1, -2, -3
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_gsm, y_gsm, z_gsm = GSEtoGSM(x, y, z, ut)
    assert np.isclose(x_gsm, -1.0, atol=1e-3)
    assert np.isclose(y_gsm, -0.540, atol=1e-3)
    assert np.isclose(z_gsm, -3.565, atol=1e-3)

def test_GSEtoGSM_large():
    x, y, z = 1e6, 2e6, 3e6
    ut = datetime.datetime(2009, 1, 27, 0, 0, 0)
    x_gsm, y_gsm, z_gsm = GSEtoGSM(x, y, z, ut)
    assert np.isclose(x_gsm, 1e6, atol=1e-3)
    assert np.isclose(y_gsm, 540626.9291896636, atol=1e-3)
    assert np.isclose(z_gsm, 3564789.2677457044, atol=1e-3)

def test_GSEtoGSM_array():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])
    ut = np.array([datetime.datetime(2009, 1, 27, 0, 0, 0),
                   datetime.datetime(2009, 1, 27, 1, 0, 0),
                   datetime.datetime(2009, 1, 27, 2, 0, 0)])
    x_gsm, y_gsm, z_gsm = GSEtoGSM(x, y, z, ut)
    assert np.allclose(x_gsm, [1, 2, 3], atol=1e-3)
    assert np.allclose(y_gsm, [0.657, 1.285, 2.0822], atol=1e-3)
    assert np.allclose(z_gsm, [8.035, 9.3461, 10.614], atol=1e-3)