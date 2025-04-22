import pytest
import numpy as np
import os
from kaipy.rcm.rcminit import LoadLAS1

# filepath: /glade/u/home/wiltbemj/src/kaipy-private/kaipy/rcm/test_rcminit.py

@pytest.fixture
def mock_las1_file(tmpdir):
    data = """# Header line 1
# Header line 2
1.0 2.0 3.0 4.0
5.0 6.0 7.0 8.0
9.0 10.0 11.0 12.0
"""
    file_path = tmpdir.join("rcmlas1")
    with open(file_path, 'w') as f:
        f.write(data)
    return str(file_path)

def test_LoadLAS1(mock_las1_file, monkeypatch):
    def mock_realpath(path):
        return os.path.dirname(mock_las1_file)
    
    monkeypatch.setattr(os.path, 'realpath', mock_realpath)
    
    alamc, etac, ikflavc, fudgec = LoadLAS1(fIn="rcmlas1")
    
    assert np.array_equal(alamc, np.array([1.0, 5.0, 9.0]))
    assert np.array_equal(etac, np.array([2.0, 6.0, 10.0]))
    assert np.array_equal(ikflavc, np.array([3.0, 7.0, 11.0]))
    assert np.array_equal(fudgec, np.array([4.0, 8.0, 12.0]))

def test_LoadLAS1_file_not_found(monkeypatch):
    def mock_realpath(path):
        return "/non/existent/path"
    
    monkeypatch.setattr(os.path, 'realpath', mock_realpath)
    
    with pytest.raises(OSError):
        LoadLAS1(fIn="nonexistentfile")

def test_LoadLAS1_invalid_format(tmpdir, monkeypatch):
    data = """# Header line 1
# Header line 2
1.0 2.0
5.0 6.0 7.0 8.0
"""
    file_path = tmpdir.join("rcmlas1")
    with open(file_path, 'w') as f:
        f.write(data)
    
    def mock_realpath(path):
        return os.path.dirname(str(file_path))
    
    monkeypatch.setattr(os.path, 'realpath', mock_realpath)
    
    with pytest.raises(ValueError):
        LoadLAS1(fIn="rcmlas1")

def test_LoadLAS1_default():
    alamc, etac, ikflavc, fudgec = LoadLAS1()
    assert len(alamc) == 90
    assert len(etac) == 90
    assert len(ikflavc) == 90
    assert len(fudgec) == 90

