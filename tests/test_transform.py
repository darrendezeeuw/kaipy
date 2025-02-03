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
