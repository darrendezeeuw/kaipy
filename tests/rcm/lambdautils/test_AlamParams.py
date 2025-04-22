import pytest
from kaipy.rcm.lambdautils.AlamParams import AlamParams, SpecParams

import kaipy.rcm.lambdautils.DistTypes as dT

def test_AlamParams_initialization():
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

    assert params.doUsePsphere is True
    assert len(params.specParams) == 2
    assert params.emine == 0.01
    assert params.eminp == 0.02
    assert params.emaxe == 10.0
    assert params.emaxp == 20.0
    assert params.L_kt == 5.0

def test_AlamParams_default_values():
    spec_params = [
        SpecParams(n=10, amin=0.1, amax=1.0, distType=dT.DistType(), flav=1)
    ]
    params = AlamParams(
        doUsePsphere=False,
        specParams=spec_params
    )

    assert params.doUsePsphere is False
    assert len(params.specParams) == 1
    assert params.emine is None
    assert params.eminp is None
    assert params.emaxe is None
    assert params.emaxp is None
    assert params.L_kt is None
