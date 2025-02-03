import pytest
import json
import numpy as np
import datetime
from kaipy.kaijson import CustomEncoder, customhook, dump, load, dumps, loads

def test_CustomEncoder_datetime():
	data = {'time': datetime.datetime(2020, 1, 1, 12, 0, 0)}
	json_str = json.dumps(data, cls=CustomEncoder)
	assert json_str == '{"time": "2020-01-01T12:00:00Z"}'

def test_CustomEncoder_numpy_array():
	data = {'array': np.array([1, 2, 3])}
	json_str = json.dumps(data, cls=CustomEncoder)
	assert json_str == '{"array": {"shape": [3], "data": [1, 2, 3]}}'

def test_CustomEncoder_numpy_float32():
	data = {'float': np.float32(1.23)}
	json_str = json.dumps(data, cls=CustomEncoder)
	assert json_str == '{"float": "_f32_1.2300000190734863"}'

def test_CustomEncoder_numpy_int64():
	data = {'int': np.int64(123)}
	json_str = json.dumps(data, cls=CustomEncoder)
	assert json_str == '{"int": "_i64_123"}'

def test_customhook_datetime():
	json_str = '{"time": "2020-01-01T12:00:00Z"}'
	data = json.loads(json_str, object_hook=customhook)
	assert data['time'] == datetime.datetime(2020, 1, 1, 12, 0, 0)

def test_customhook_numpy_array():
	json_str = '{"array": {"shape": [3], "data": [1, 2, 3]}}'
	data = json.loads(json_str, object_hook=customhook)
	assert np.array_equal(data['array'], np.array([1, 2, 3]))

def test_customhook_numpy_float32():
	json_str = '{"float": "_f32_1.2300000190734863"}'
	data = json.loads(json_str, object_hook=customhook)
	assert data['float'] == np.float32(1.23)

def test_customhook_numpy_int64():
	json_str = '{"int": "_i64_123"}'
	data = json.loads(json_str, object_hook=customhook)
	assert data['int'] == np.int64(123)

def test_dump_load(tmpdir):
	data = {
		'time': datetime.datetime(2020, 1, 1, 12, 0, 0),
		'array': np.array([1, 2, 3]),
		'float': np.float32(1.23),
		'int': np.int64(123)
	}
	file_path = tmpdir.join("test.json")
	dump(str(file_path), data)
	loaded_data = load(str(file_path))
	assert loaded_data['time'][0] == datetime.datetime(2020, 1, 1, 12, 0, 0)
	assert np.array_equal(loaded_data['array'], np.array([1, 2, 3]))
	assert loaded_data['float'] == np.float32(1.23)
	assert loaded_data['int'] == np.int64(123)

def test_dumps():
	data = {
		'time': datetime.datetime(2020, 1, 1, 12, 0, 0),
		'array': np.array([1, 2, 3]),
		'float': np.float32(1.23),
		'int': np.int64(123)
	}
	json_str = dumps(data)
	expected_str = (
		'{\n'
		'    "time": "2020-01-01T12:00:00Z",\n'
		'    "array": {\n'
		'        "shape": [\n'
		'            3\n'
		'        ],\n'
		'        "data": [\n'
		'            1,\n'
		'            2,\n'
		'            3\n'
		'        ]\n'
		'    },\n'
		'    "float": "_f32_1.2300000190734863",\n'
		'    "int": "_i64_123"\n'
		'}'
	)
	assert json_str == expected_str

def test_loads():
	json_str = (
		'{\n'
		'    "time": "2020-01-01T12:00:00Z",\n'
		'    "array": {\n'
		'        "shape": [\n'
		'            3\n'
		'        ],\n'
		'        "data": [\n'
		'            1,\n'
		'            2,\n'
		'            3\n'
		'        ]\n'
		'    },\n'
		'    "float": "_f32_1.2300000190734863",\n'
		'    "int": "_i64_123"\n'
		'}'
	)
	data = loads(json_str)
	assert data['time'][0] == datetime.datetime(2020, 1, 1, 12, 0, 0)
	assert np.array_equal(data['array'], np.array([1, 2, 3]))
	assert data['float'] == np.float32(1.23)
	assert data['int'] == np.int64(123)