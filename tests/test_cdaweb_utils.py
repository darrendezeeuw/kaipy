import pytest
import datetime
import numpy as np
from unittest.mock import patch, MagicMock

from kaipy.cdaweb_utils import (
    fetch_satellite_geographic_position,
    fetch_satellite_SM_position,
    fetch_spacecraft_SM_trajectory,
    fetch_satellite_magnetic_northern_footprint_position,
    fetch_satellite_magnetic_southern_footprint_position,
    fetch_helio_spacecraft_HGS_trajectory,
    fetch_helio_spacecraft_trajectory,
    ingest_helio_spacecraft_trajectory
)

@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_satellite_geographic_position(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'Ephem': {
                'Id': 'test_dataset',
                'Data': 'test_variable',
                'CoordSys': 'GEO'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {'test_variable': np.array([1, 2, 3])})

    when = datetime.datetime(2020, 1, 1, 0, 0, 0)
    sc_rad, sc_lat, sc_lon = fetch_satellite_geographic_position('test_spacecraft', when)
    assert np.isclose(sc_rad, 3.7416, atol=1e-3)
    assert np.isclose(sc_lat, 53.3007, atol=1e-3)
    assert np.isclose(sc_lon, 63.4349, atol=1e-3)

@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_satellite_SM_position(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'Ephem': {
                'Id': 'test_dataset',
                'Data': 'test_variable',
                'CoordSys': 'GSM'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {'test_variable': np.array([1, 2, 3])})

    when = datetime.datetime(2020, 1, 1, 0, 0, 0)
    sc_x, sc_y, sc_z = fetch_satellite_SM_position('test_spacecraft', when)
    assert np.isclose(sc_x, 2.1907, atol=1e-3)
    assert np.isclose(sc_y, 2.0, atol=1e-3)
    assert np.isclose(sc_z, 2.2805, atol=1e-3)


@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_spacecraft_SM_trajectory(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'Ephem': {
                'Id': 'test_dataset',
                'Data': 'test_variable',
                'CoordSys': 'GSM'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {'test_variable': np.array([[1, 2, 3], [4, 5, 6]]),
                                               'Epoch': np.array([datetime.datetime(2020, 1, 1, 0, 0, 0),
                                                                  datetime.datetime(2020, 1, 1, 1, 0, 0)])})

    t_start = datetime.datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime.datetime(2020, 1, 1, 1, 0, 0)
    x, y, z = fetch_spacecraft_SM_trajectory('test_spacecraft', t_start, t_end)
    assert np.allclose(x, [2.1907, 6.3311], atol=1e-3)
    assert np.allclose(y, [2.0, 5.0], atol=1e-3)
    assert np.allclose(z, [2.2805, 3.4520], atol=1e-3)    

@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_satellite_magnetic_northern_footprint_position(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'MagneticFootprintNorth': {
                'Id': 'test_dataset',
                'Data': ['lat_variable', 'lon_variable'],
                'CoordSys': 'GEO'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {'lat_variable': np.array([10]), 'lon_variable': np.array([20])})

    when = datetime.datetime(2020, 1, 1, 0, 0, 0)
    fp_lat, fp_lon = fetch_satellite_magnetic_northern_footprint_position('test_spacecraft', when)

    assert fp_lat == 10
    assert fp_lon == 20

@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_satellite_magnetic_southern_footprint_position(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'MagneticFootprintSouth': {
                'Id': 'test_dataset',
                'Data': ['lat_variable', 'lon_variable'],
                'CoordSys': 'GEO'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {'lat_variable': np.array([30]), 'lon_variable': np.array([40])})

    when = datetime.datetime(2020, 1, 1, 0, 0, 0)
    fp_lat, fp_lon = fetch_satellite_magnetic_southern_footprint_position('test_spacecraft', when)

    assert fp_lat == 30
    assert fp_lon == 40

@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_helio_spacecraft_HGS_trajectory(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'Ephem': {
                'Id': 'test_dataset',
                'Data': {
                    'HGI_LON': 'HGI_LON',
                    'HGI_LAT': 'HGI_LAT',
                    'RAD_AU': 'RAD_AU'
                },
                'CoordSys': 'HGI'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {
        'HGI_LON': np.array([10, 20]),
        'HGI_LAT': np.array([30, 40]),
        'RAD_AU': np.array([1, 2]),
        'Epoch': np.array([datetime.datetime(2020, 1, 1, 0, 0, 0), datetime.datetime(2020, 1, 1, 1, 0, 0)])
    })

    t_start = datetime.datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime.datetime(2020, 1, 1, 1, 0, 0)
    mjdc = 58849.0
    x, y, z = fetch_helio_spacecraft_HGS_trajectory('test_spacecraft', t_start, t_end, mjdc)

    assert np.allclose(x, [180.844, 328.722], atol=1e-3)
    assert np.allclose(y, [-44.433, -21.857], atol=1e-3)
    assert np.allclose(z, [107.516, 276.440], atol=1e-3)

@patch('kaipy.cdaweb_utils.scutils.getScIds')
@patch('kaipy.cdaweb_utils.CdasWs')
def test_fetch_helio_spacecraft_trajectory(mock_CdasWs, mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'Ephem': {
                'Id': 'test_dataset',
                'Data': 'test_variable',
                'CoordSys': 'HGI'
            }
        }
    }
    mock_cdas = mock_CdasWs.return_value
    mock_cdas.get_data.return_value = (200, {'test_variable': np.array([[1, 2, 3], [4, 5, 6]])})

    t_start = datetime.datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime.datetime(2020, 1, 1, 1, 0, 0)
    data = fetch_helio_spacecraft_trajectory('test_spacecraft', t_start, t_end)

    assert np.array_equal(data['test_variable'], [[1, 2, 3], [4, 5, 6]])

@patch('kaipy.cdaweb_utils.scutils.getScIds')
def test_ingest_helio_spacecraft_trajectory(mock_getScIds):
    mock_getScIds.return_value = {
        'test_spacecraft': {
            'Ephem': {
                'CoordSys': 'HGI',
                'Data': {
                    'HGI_LAT': 'HGI_LAT',
                    'HGI_LON': 'HGI_LON',
                    'RAD_AU': 'RAD_AU'
                }
            }
        }
    }
    sc_data = {
        'HGI_LAT': np.array([10, 20]),
        'HGI_LON': np.array([30, 40]),
        'RAD_AU': np.array([1, 2]),
        'Epoch': np.array([datetime.datetime(2020, 1, 1, 0, 0, 0), datetime.datetime(2020, 1, 1, 1, 0, 0)])
    }
    MJDc = 58849.0
    x, y, z = ingest_helio_spacecraft_trajectory('test_spacecraft', sc_data, MJDc)
    assert np.allclose(x, [-210.528, -388.089], atol=1e-3)
    assert np.allclose(y, [-22.855, -112.720], atol=1e-3)
    assert np.allclose(z, [37.340, 147.090], atol=1e-3)