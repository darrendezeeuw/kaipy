import pytest
import numpy as np
from kaipy.rcm.lambdautils.genAlam import getAlamMinMax, genSpeciesFromParams, genAlamDataFromParams, genPsphereSpecies
from kaipy.rcm.lambdautils.AlamParams import SpecParams, AlamParams
from kaipy.rcm.lambdautils.AlamData import Species, AlamData
import kaipy.rcm.lambdautils.DistTypes as dT

def setup_AlamParams():
    wolfP1 = 3
    wolfP2 = 1
    spec_params = [
        SpecParams(n=10, amin=0.1, amax=1.0, distType=dT.DT_Wolf(p1=wolfP1,p2=wolfP2), flav=1),
        SpecParams(n=20, amin=0.2, amax=2.0, distType=dT.DT_Wolf(p1=wolfP1,p2=wolfP2), flav=2)
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

def test_getAlamMinMax():
    alams = [1, 2, 3, 4]
    amin, amax = getAlamMinMax(alams)
    assert amin == [0, 1.5, 2.5, 3.5]
    assert amax == [1.5, 2.5, 3.5, 4.5]

def test_genSpeciesFromParams():
    specParams = SpecParams(n=10, amin=0.1, amax=1.0, distType=dT.DistType(), flav=1)
    specParams.genAlams = lambda: [1, 2, 3]
    specParams.flav = 'test_flav'
    specParams.fudge = 0.1
    specParams.name = 'test_name'
    
    species = genSpeciesFromParams(specParams)
    
    assert isinstance(species, Species)
    assert species.n == 3
    assert species.alams == [1, 2, 3]
    assert species.amins == [0, 1.5, 2.5]
    assert species.amaxs == [1.5, 2.5, 3.5]
    assert species.flav == 'test_flav'
    assert species.fudge == 0.1
    assert species.name == 'test_name'

def test_genAlamDataFromParams():
    params = setup_AlamParams()
    
    alamData = genAlamDataFromParams(params)
    
    assert alamData.doUsePsphere == True
    assert len(alamData.specs) == 3
    assert isinstance(alamData.specs[0], Species)
    assert alamData.specs[0].name == "Plasmasphere"
    assert isinstance(alamData.params, AlamParams)

def test_genPsphereSpecies():
    species = genPsphereSpecies()
    
    assert isinstance(species, Species)
    assert species.n == 1
    assert species.alams == [0]
    assert species.amins == [0]
    assert species.amaxs == [0]
    assert species.flav == 1
    assert species.fudge == 0
    assert species.name == 'Plasmasphere'