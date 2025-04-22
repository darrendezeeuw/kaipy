import pytest
import numpy as np
import h5py
from kaipy.rcm.rcmutils import getSpecieslambdata, getClosedRegionMask, getCumulPress, getValAtLoc_linterp, RCMSpeciesInfo, RCMInfo


def test_getSpecieslambdata():
    rcmS0 = {
        'alamc': np.array([0, 1, 2, 3, 4])
    }
    result = getSpecieslambdata(rcmS0, species='ions')
    assert result['kStart'] == 1
    assert result['kEnd'] == 5
    assert np.array_equal(result['ilamc'], np.array([1, 2, 3, 4]))
    assert np.array_equal(result['ilami'], np.array([0, 1.5, 2.5, 3.5, 4.5]))
    assert np.array_equal(result['lamscl'], np.diff(result['ilami']) * np.sqrt(result['ilamc']))

def test_getClosedRegionMask():
    s5 = {
        'rcmxmin': np.array([0.5, 1.5]),
        'rcmymin': np.array([0.5, 1.5]),
        'rcmvm': np.array([0, 1])
    }
    mask = getClosedRegionMask(s5)
    assert np.array_equal(mask, np.array([True, False]))

def test_getCumulPress():
    ilamc = np.array([1, 2, 3])
    vm = np.array([[1, 2], [3, 4]])
    eetas = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    pCumul = getCumulPress(ilamc, vm, eetas)
    assert pCumul.shape == (3, 2, 2)

def test_getValAtLoc_linterp():
    xData = [1, 2, 3, 4]
    yData = [2, 4, 6, 8]
    val = 2.5
    result = getValAtLoc_linterp(val, xData, yData, getAxis='y')
    assert result == 5.0
    
def test_RCMSpeciesInfo_initialization():
    N = 5
    flav = 2
    kStart = 0
    kEnd = 5
    alamc = np.array([0, 1, 2, 3, 4])
    species_info = RCMSpeciesInfo(N, flav, kStart, kEnd, alamc)

    assert species_info.N == N
    assert species_info.flav == flav
    assert species_info.kStart == kStart
    assert species_info.kEnd == kEnd
    assert np.array_equal(species_info.alamc, alamc)
    assert np.array_equal(species_info.alami, np.array([0, 0.5, 1.5, 2.5, 3.5, 4.5]))

def test_RCMSpeciesInfo_single_cell():
    N = 1
    flav = 1
    kStart = 0
    kEnd = 1
    alamc = np.array([0])
    species_info = RCMSpeciesInfo(N, flav, kStart, kEnd, alamc)

    assert species_info.N == N
    assert species_info.flav == flav
    assert species_info.kStart == kStart
    assert species_info.kEnd == kEnd
    assert np.array_equal(species_info.alamc, alamc)
    assert np.array_equal(species_info.alami, np.array([0, 0]))




