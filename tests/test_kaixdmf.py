import pytest
import numpy as np
import h5py
from kaipy.kaixdmf import AddGrid, AddData, AddDI, getRootVars, getVars, printVidAndLocs, AddVectors, getLoc, addHyperslab

import xml.etree.ElementTree as et

def test_AddGrid():
	Geom = et.Element("Geometry")
	AddGrid("test.h5", Geom, "3 3 3", ["X", "Y", "Z"])
	assert len(Geom.findall("DataItem")) == 3
	for i, coord in enumerate(["X", "Y", "Z"]):
		data_item = Geom.findall("DataItem")[i]
		assert data_item.get("Dimensions") == "3 3 3"
		assert data_item.get("NumberType") == "Float"
		assert data_item.get("Precision") == "4"
		assert data_item.get("Format") == "HDF"
		assert data_item.text == f"test.h5:/{coord}"

def test_AddData():
	Grid = et.Element("Grid")
	AddData(Grid, "test.h5", "density", "Cell", "3 3 3", 0)
	vAtt = Grid.find("Attribute")
	assert vAtt.get("Name") == "density"
	assert vAtt.get("AttributeType") == "Scalar"
	assert vAtt.get("Center") == "Cell"
	aDI = vAtt.find("DataItem")
	assert aDI.get("Dimensions") == "3 3 3"
	assert aDI.get("NumberType") == "Float"
	assert aDI.get("Precision") == "4"
	assert aDI.get("Format") == "HDF"
	assert aDI.text == "test.h5:/Step#0/density"

def test_AddDI():
	elt = et.Element("Element")
	AddDI(elt, "test.h5", 0, "3 3 3", "density")
	aDI = elt.find("DataItem")
	assert aDI.get("Dimensions") == "3 3 3"
	assert aDI.get("NumberType") == "Float"
	assert aDI.get("Precision") == "4"
	assert aDI.get("Format") == "HDF"
	assert aDI.text == "test.h5:/Step#0/density"

def test_getRootVars(tmpdir):
	fname = tmpdir.join("test.h5")
	with h5py.File(fname, 'w') as hf:
		hf.create_dataset("density", data=np.random.rand(3, 3, 3))
		hf.create_dataset("X", data=np.random.rand(3))
		hf.create_dataset("Y", data=np.random.rand(3))
		hf.create_dataset("Z", data=np.random.rand(3))
	vIds, vLocs = getRootVars(str(fname), [3, 3, 3])
	assert "density" in vIds
	assert "X" not in vIds
	assert "Y" not in vIds
	assert "Z" not in vIds

def test_getVars(tmpdir):
	fname = tmpdir.join("test.h5")
	with h5py.File(fname, 'w') as hf:
		grp = hf.create_group("Step#0")
		grp.create_dataset("density", data=np.random.rand(3, 3, 3))
	vIds, vLocs = getVars(str(fname), "Step#0", [3, 3, 3])
	assert "density" in vIds

def test_printVidAndLocs(capsys):
	vIds = ["density", "pressure"]
	vLocs = ["Cell", "Node"]
	printVidAndLocs(vIds, vLocs)
	captured = capsys.readouterr()
	assert "density\t: Cell" in captured.out
	assert "pressure\t: Node" in captured.out

def test_AddVectors():
	Grid = et.Element("Grid")
	AddVectors(Grid, "test.h5", ["Vx", "Vy"], "3 3 3", "3 3 3", 2, 0)
	vAtt = Grid.find("Attribute")
	assert vAtt.get("Name") == "VecV"
	assert vAtt.get("AttributeType") == "Vector"
	assert vAtt.get("Center") == "Cell"
	fDI = vAtt.find("DataItem")
	assert fDI.get("ItemType") == "Function"
	assert fDI.get("Dimensions") == "3 3 3"
	assert fDI.get("Function") == "JOIN($0 , $1)"
	data_items = fDI.findall("DataItem")
	assert len(data_items) == 2
	assert data_items[0].text == "test.h5:/Step#0/Vx"
	assert data_items[1].text == "test.h5:/Step#0/Vy"

def test_getLoc():
	assert getLoc([4, 4, 4], [3, 3, 3]) == "Cell"
	assert getLoc([3, 3, 3], [3, 3, 3]) == "Node"
	assert getLoc([3, 3, 3], [5, 5, 5]) == "Other"

def test_addHyperslab():
	Grid = et.Element("Grid")
	addHyperslab(Grid, "density", "3 3 3", "3 3 3", "0 0 0", "1 1 1", "3 3 3", "3 3 3", "test.h5:/density")
	vAtt = Grid.find("Attribute")
	assert vAtt.get("Name") == "density"
	assert vAtt.get("AttributeType") == "Scalar"
	assert vAtt.get("Center") == "Node"
	slabDI = vAtt.find("DataItem")
	assert slabDI.get("ItemType") == "HyperSlab"
	assert slabDI.get("Dimensions") == "3 3 3"
	cutDI = slabDI.find("DataItem")
	assert cutDI.get("Dimensions") == "3 3 3"
	assert cutDI.get("Format") == "XML"
	assert cutDI.text == "\n0 0 0\n1 1 1\n3 3 3\n"
