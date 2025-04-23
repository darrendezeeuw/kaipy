import pytest
from kaipy.rcm.lambdautils.AlamData import Species, AlamData
from kaipy.rcm.lambdautils.AlamParams import AlamParams, SpecParams
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

def test_species_initialization():
    params = setup_AlamParams()
    species = Species(
        n=3,
        alams=[1.0, 2.0, 3.0],
        amins=[0.5, 1.5, 2.5],
        amaxs=[1.5, 2.5, 3.5],
        flav=1,
        fudge=0.1,
        params=params,
        name="TestSpecies"
    )

    
    assert species.n == 3
    assert species.alams == [1.0, 2.0, 3.0]
    assert species.amins == [0.5, 1.5, 2.5]
    assert species.amaxs == [1.5, 2.5, 3.5]
    assert species.flav == 1
    assert species.fudge == 0.1
    assert isinstance(species.params, AlamParams)
    assert species.name == "TestSpecies"

def test_alamdata_initialization():
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

    alam_data = AlamData(
        doUsePsphere=True,
        specs=species_list,
        params=params
    )

    assert alam_data.doUsePsphere is True
    assert len(alam_data.specs) == 2
    assert isinstance(alam_data.specs[0], Species)
    assert alam_data.specs[0].name == "TestSpecies1"
    assert alam_data.specs[1].name == "TestSpecies2"
    assert isinstance(alam_data.params, AlamParams)