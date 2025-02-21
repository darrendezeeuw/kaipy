import pytest
import numpy as np
from kaipy.rcm.wmutils.genWM import genWM, genh5, readPoly, ChorusPoly, ReSample, genChorus
from kaipy.rcm.wmutils.wmData import wmParams


def test_ChorusPoly_basic():
        Li = np.array([3.0, 4.0, 5.0])
        Eki = np.array([0.001, 0.01, 0.1])
        polyArray = np.random.rand(7, 24, 10)  # Mock polynomial coefficients

        result = ChorusPoly(Li, Eki, polyArray)

        assert result.shape == (7, 24, 3, 3)
        assert np.all(result > 0)

def test_ChorusPoly_zero_coefficients():
        Li = np.array([3.0, 4.0, 5.0])
        Eki = np.array([0.001, 0.01, 0.1])
        polyArray = np.zeros((7, 24, 10))  # All coefficients are zero

        result = ChorusPoly(Li, Eki, polyArray)

        assert result.shape == (7, 24, 3, 3)
        assert np.all(result == 86400)  # 10^0 * (60 * 60 * 24) = 86400

def test_ChorusPoly_specific_coefficients():
        Li = np.array([3.0, 4.0, 5.0])
        Eki = np.array([0.001, 0.01, 0.1])
        polyArray = np.zeros((7, 24, 10))
        polyArray[:, :, 0] = 1  # Set intercept to 1

        result = ChorusPoly(Li, Eki, polyArray)

        assert result.shape == (7, 24, 3, 3)
        assert np.all(result == 864000)  # 10^1 * (60 * 60 * 24) = 864000

def test_ChorusPoly_varying_coefficients():
        Li = np.array([3.0, 4.0, 5.0])
        Eki = np.array([0.001, 0.01, 0.1])
        polyArray = np.zeros((7, 24, 10))
        polyArray[:, :, 1] = 1  # Set L coefficient to 1

        result = ChorusPoly(Li, Eki, polyArray)

        assert result.shape == (7, 24, 3, 3)
        assert np.all(result > 86400)  # Result should be greater than 86400 due to L coefficient

def test_ChorusPoly_large_values():
        Li = np.array([3.0, 4.0, 5.0])
        Eki = np.array([0.001, 0.01, 0.1])
        polyArray = np.ones((7, 24, 10)) * 100  # Large coefficients

        result = ChorusPoly(Li, Eki, polyArray)

        assert result.shape == (7, 24, 3, 3)
        assert np.all(result > 0)

def test_genWM_basic():
        params = wmParams()
        result = genWM(params)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[0].shape == (6,)
        assert result[1].shape == (97,)
        assert result[2].shape == (41,)
        assert result[3].shape == (155,)
        assert result[4].shape == (155, 41, 97, 6)
