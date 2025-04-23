import pytest
import numpy as np
import h5py
import os

import kaipy.rcm.lambdautils.AlamParams as aP
from kaipy.rcm.lambdautils.fileIO import saveRCMConfig, saveParams, loadParams, saveData
from kaipy.rcm.lambdautils.AlamParams import AlamParams, SpecParams
from kaipy.rcm.lambdautils.AlamData import Species, AlamData
import kaipy.rcm.lambdautils.DistTypes as dT

def setup_AlamParams():
    spec_params = [
        SpecParams(n=10, amin=0.1, amax=1.0, distType=dT.DistType(), flav=1),
        SpecParams(n=20, amin=0.2, amax=2.0, distType=dT.DistType(), flav=2)
    ]
    params = AlamParams(
        doUsePsphere=True,
        specParams=spec_params,
        emine=0.01,
        eminp=0.02,
        emaxe=10.0,
        emaxp=20.0,
        L_kt=5.0
    )
    return params

def setup_AlamData():
    params = setup_AlamParams()
    species_list = [
        Species(
            n=3,
            alams=[1.0, 2.0, 3.0],
            amins=[0.5, 1.5, 2.5],
            amaxs=[1.5, 2.5, 3.5],
            flav=1,
            fudge=0.1,
            name="TestSpecies1"
        ),
        Species(
            n=2,
            alams=[4.0, 5.0],
            amins=[3.5, 4.5],
            amaxs=[4.5, 5.5],
            flav=2,
            fudge=0.2,
            name="TestSpecies2"
        )
    ]
    alamData = AlamData(
        doUsePsphere=True,
        specs=species_list,
        params=params
    )
    return alamData

def test_saveRCMConfig(tmp_path):
    alamData = setup_AlamData()
    fname = tmp_path / 'rcmconfig.h5'
    saveRCMConfig(alamData, params=None, fname=fname)
    assert os.path.exists(fname)

def test_saveParams(tmp_path):
    f5 = h5py.File(tmp_path / 'testfile.h5', 'w')
    alamParams = setup_AlamParams()
    saveParams(f5, alamParams)
    assert 'AlamParams' in f5.attrs
    f5.close()

def test_loadParams(tmp_path):
    f5 = h5py.File(tmp_path / 'testfile.h5', 'w')
    alamParams = setup_AlamParams()
    saveParams(f5, alamParams)
    f5.close()

    f5 = h5py.File(tmp_path / 'testfile.h5', 'r')
    loaded_params = loadParams(f5)
    print(f'loaded_params: {loaded_params}')
    print(f'alamParams: {alamParams}')
    assert loaded_params['doUsePsphere'] == alamParams.doUsePsphere

    f5.close()
