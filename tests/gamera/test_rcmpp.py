import pytest
import numpy as np
from kaipy.gamera.rcmpp import RCMEq

# filepath: /glade/u/home/wiltbemj/src/kaipy-private/kaipy/gamera/test_rcmpp.py
import numpy.ma as ma

class MockRCMData:
    def __init__(self, xMin, yMin, zMin=None, IOpen=None):
        self.data = {
            "xMin": xMin,
            "yMin": yMin,
            "zMin": zMin,
            "IOpen": IOpen
        }

    def GetVar(self, var_name, nStp, doVerb=True):
        return self.data[var_name]

@pytest.fixture
def rcmdata():
    xMin = np.array([1.0, 2.0, 3.0])
    yMin = np.array([4.0, 5.0, 6.0])
    zMin = np.array([7.0, 8.0, 9.0])
    IOpen = np.array([0, 1, 0])
    return MockRCMData(xMin, yMin, zMin, IOpen)

def test_RCMEq_no_mask_no_xyz(rcmdata):
    bmX, bmY = RCMEq(rcmdata, nStp=0, doMask=False, doXYZ=False, doVerb=False)
    assert np.array_equal(bmX, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(bmY, np.array([4.0, 5.0, 6.0]))

def test_RCMEq_no_mask_xyz(rcmdata):
    bmX, bmY = RCMEq(rcmdata, nStp=0, doMask=False, doXYZ=True, doVerb=False)
    expected_bmX = np.array([1.97036873, 3.58156198, 5.01996016])
    expected_bmY = np.array([ 7.88147493,  8.95390495, 10.03992032])
    assert np.allclose(bmX, expected_bmX)
    assert np.allclose(bmY, expected_bmY)

def test_RCMEq_mask_no_xyz(rcmdata):
    bmX, bmY = RCMEq(rcmdata, nStp=0, doMask=True, doXYZ=False, doVerb=False)
    expected_bmX = ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
    expected_bmY = ma.masked_array([4.0, 5.0, 6.0], mask=[False, True, False])
    assert np.array_equal(bmX, expected_bmX)
    assert np.array_equal(bmY, expected_bmY)

def test_RCMEq_mask_xyz(rcmdata):
    bmX, bmY = RCMEq(rcmdata, nStp=0, doMask=True, doXYZ=True, doVerb=False)
    expected_bmX = ma.masked_array([8.06225775, 9.43398113, 10.81665383], mask=[True, True, True])
    expected_bmY = ma.masked_array([4.0, 5.0, 6.0], mask=[True, True, True])
    assert np.array_equal(bmX.mask, expected_bmX.mask)
    assert np.array_equal(bmY.mask, expected_bmY.mask)