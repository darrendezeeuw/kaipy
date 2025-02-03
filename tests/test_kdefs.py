import pytest
import numpy as np
from kaipy.kdefs import *

def test_helpful_conversions():
    assert G2nT == 1E5
    assert kev2J == 1.602176634E-16
    assert ev2J == 1.602176634E-19
    assert erg2J == 1e-7

def test_physical_constants():
    assert Mu0 == 4E-7 * np.pi
    assert Me_cgs == 9.1093837015E-28
    assert Mp_cgs == 1.67262192369E-24
    assert eCharge == 1.602E-19
    assert dalton == 1.66053906660 * 1.0E-27

def test_planetary_constants():
    assert Re_cgs == 6.3781E8
    assert EarthM0g == 0.2961737
    assert REarth == Re_cgs * 1.0e-2
    assert RionE == 6.5
    assert EarthPsi0 == 92.4
    assert SaturnM0g == 0.21
    assert RSaturnXE == 9.5
    assert JupiterM0g == 4.8
    assert RJupiterXE == 11.209
    assert MercuryM0g == 0.00345
    assert RMercuryXE == 0.31397
    assert NeptuneM0g == 0.142
    assert RNeptuneXE == 3.860

def test_helio_constants():
    assert Rsolar == 6.957e10
    assert kbltz == 1.38e-16
    assert mp == 1.67e-24
    assert Tsolar == 25.38
    assert Tsolar_synodic == 27.28
    assert JD2MJD == 2400000.5
    assert Day2s == 86400.
    assert vc_cgs == 2.99792458e10

def test_output_defaults():
    assert barLen == 30
    assert barLab == 30


def test_io_defaults():
    assert grpTimeCache == "timeAttributeCache"