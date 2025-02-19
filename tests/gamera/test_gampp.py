import pytest
import numpy as np
import h5py
from astropy.time import Time
import datetime
from kaipy.gamera.gampp import GameraPipe
from kaipy.kaiH5 import PullVar, PullVarLoc, cntSteps, getTs, getDims, getRootVars, getVars
from tests.conftest import Ni, Nj, Nk  # Import the constants

def test_open_pipe(gamera_pipe):
    assert gamera_pipe.fdir.endswith("data")
    assert gamera_pipe.ftag == "test.gam"
    assert gamera_pipe.isMPI == False
    assert gamera_pipe.Nt == 3
    assert np.array_equal(gamera_pipe.sids, np.array([0, 1, 2]))

def test_set_units(gamera_pipe):
    print(f'test_gampp: {gamera_pipe.UnitsID}')
    gamera_pipe.SetUnits(gamera_pipe.f0)
    print(f'test_gampp: {gamera_pipe.UnitsID}')
    assert gamera_pipe.UnitsID == "CODE"

def test_get_grid(gamera_pipe):
    gamera_pipe.GetGrid(doVerbose=False)
    assert gamera_pipe.gridLoaded == True
    print(gamera_pipe.X.shape)
    assert gamera_pipe.X.shape == (Ni+1, Nj+1, Nk+1)
    assert gamera_pipe.Y.shape == (Ni+1, Nj+1, Nk+1)
    assert gamera_pipe.Z.shape == (Ni+1, Nj+1, Nk+1)

def test_get_var(gamera_pipe):
    V = gamera_pipe.GetVar("D", sID=0)
    assert V.shape == (Ni, Nj, Nk)
    assert np.array_equal(V, np.zeros((Ni, Nj, Nk)))

def test_get_slice(gamera_pipe):
    Vs = gamera_pipe.GetSlice("D", sID=0, ijkdir='idir', n=1)
    assert Vs.shape == (Nj, Nk)
    assert np.array_equal(Vs, np.zeros((Nj, Nk)))

def test_get_root_var(gamera_pipe):
    V = gamera_pipe.GetRootVar("dV")
    assert V.shape == (Ni, Nj, Nk)
    assert np.array_equal(V, np.zeros((Ni, Nj, Nk)))

def test_get_root_slice(gamera_pipe):
    Vs = gamera_pipe.GetRootSlice("dV", ijkdir='idir', n=1)
    assert Vs.shape == (Nj, Nk)
    assert np.array_equal(Vs, np.zeros((Nj, Nk)))