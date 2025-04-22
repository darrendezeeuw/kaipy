import pytest
from kaipy.rcm.wmutils.wmData import wmParams

def test_wmParams_initialization():
    params = wmParams(dim=5, nKp=7, nMLT=36, nL=40, nEk=100)
    
    assert params.dim == 5
    assert params.nKp == 7
    assert params.nMLT == 36
    assert params.nL == 40
    assert params.nEk == 100

def test_wmParams_default_initialization():
    params = wmParams()
    
    assert params.dim == 4
    assert params.nKp == 6
    assert params.nMLT == 97
    assert params.nL == 41
    assert params.nEk == 155

def test_wmParams_getAttrs():
    params = wmParams(dim=4, nKp=6, nMLT=97, nL=41, nEk=155)
    attrs = params.getAttrs()
    
    assert attrs['tauDim'] == 4
    assert attrs['nKp'] == 6
    assert attrs['nMLT'] == 97
    assert attrs['nL'] == 41
    assert attrs['nEk'] == 155