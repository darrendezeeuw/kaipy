import pytest
import numpy as np
import h5py
import datetime
from astropy.time import Time
from kaipy.gamera.magsphere import GamsphPipe
from kaipy.gamera.gampp import GameraPipe
import kaipy.kdefs as kdefs


def test_initialization(gamera_pipe):
    assert gamera_pipe.MagM == -1.0 * kdefs.EarthM0g * 1.0e5
    assert gamera_pipe.bScl == 4.58
    assert gamera_pipe.pScl == 1.67e-2
    assert gamera_pipe.vScl == 1.0e+2
    assert gamera_pipe.tScl == 63.8
    assert gamera_pipe.hasRemix == True
    assert isinstance(gamera_pipe.mixPipe, GameraPipe)
    assert gamera_pipe.hasRemixO == False
    assert gamera_pipe.Nrm == 0

def test_open_pipe(gamera_pipe):
    gamera_pipe.OpenPipe(doVerbose=False)
    assert gamera_pipe.bScl == 4.58
    assert gamera_pipe.pScl == 1.67e-2
    assert gamera_pipe.vScl == 1.0e+2
    assert gamera_pipe.tScl == 63.8

def test_get_m0(gamera_pipe):
    gamera_pipe.GetM0()
    assert gamera_pipe.MagM == -1.0 * kdefs.EarthM0g * 1.0e5

def test_egg_slice(gamera_pipe):
    gamera_pipe.OpenPipe(doVerbose=False)
    Qk = gamera_pipe.EggSlice("D", sID=0)
    assert Qk.shape == (gamera_pipe.Ni, 2 * gamera_pipe.Nj)

def test_del_bz(gamera_pipe):
    gamera_pipe.OpenPipe(doVerbose=False)
    dbz = gamera_pipe.DelBz(s0=0)
    assert dbz.shape == (gamera_pipe.Ni, 2 * gamera_pipe.Nj)

def test_eq_mag_b(gamera_pipe):
    gamera_pipe.OpenPipe(doVerbose=False)
    Beq = gamera_pipe.eqMagB(s0=0)
    assert Beq.shape == (gamera_pipe.Ni, 2 * gamera_pipe.Nj)

# def test_b_stream(gamera_pipe):
#     gamera_pipe.OpenPipe(doVerbose=False)
#     x1, y1, gu, gv, gM = gamera_pipe.bStream(s0=0)
#     assert x1.shape[0] > 0
#     assert y1.shape[0] > 0
#     assert gu.shape == gv.shape == gM.shape

# def test_v_stream(gamera_pipe):
#     gamera_pipe.OpenPipe(doVerbose=False)
#     x1, y1, gu, gv, gM = gamera_pipe.vStream(s0=0)
#     assert x1.shape[0] > 0
#     assert y1.shape[0] > 0
#     assert gu.shape == gv.shape == gM.shape

def test_gam2remix(gamera_pipe):
    gamera_pipe.OpenPipe(doVerbose=False)
    fMix = gamera_pipe.Gam2Remix(0)
    assert fMix is None

def test_get_cpcp(gamera_pipe):
    gamera_pipe.OpenPipe(doVerbose=False)
    cpcp = gamera_pipe.GetCPCP(0)
    assert cpcp == [0.0, 0.0]

def test_add_time(gamera_pipe):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    gamera_pipe.OpenPipe(doVerbose=False)
    gamera_pipe.AddTime(0, ax)
    assert len(ax.texts) > 0

def test_add_sw(gamera_pipe):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    gamera_pipe.OpenPipe(doVerbose=False)
    gamera_pipe.AddSW(0, ax)
    assert len(ax.texts) > 0

def test_add_cpcp(gamera_pipe):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    gamera_pipe.OpenPipe(doVerbose=False)
    gamera_pipe.AddCPCP(0, ax)
    assert len(ax.texts) > 0

# def test_do_stream(gamera_pipe):
#     gamera_pipe.OpenPipe(doVerbose=False)
#     U = np.random.rand(gamera_pipe.Ni, 2 * gamera_pipe.Nj)
#     V = np.random.rand(gamera_pipe.Ni, 2 * gamera_pipe.Nj)
#     x1, y1, gu, gv, gM = gamera_pipe.doStream(U, V)
#     assert x1.shape[0] > 0
#     assert y1.shape[0] > 0
#     assert gu.shape == gv.shape == gM.shape

# def test_cmiviz(gamera_pipe):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     gamera_pipe.OpenPipe(doVerbose=False)
#     gamera_pipe.CMIViz(AxM=ax, nStp=0)
#     assert len(ax.images) > 0