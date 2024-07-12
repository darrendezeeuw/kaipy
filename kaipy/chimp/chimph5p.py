#Various routines to deal with h5p data
import numpy as np
import h5py

import kaipy.kaiH5 as kH5

#Count number of particles in an h5p file
def cntTPs(fname):
	"""
	Count the number of particles in the given H5Part file.

	Args:
		fname (str): The path to the H5Part file.

	Returns:
		int: The number of particles in the H5Part file.
	"""
	with h5py.File(fname, 'r') as hf:
		grp = hf.get("Step#0")
		ids = grp.get("id")[()]
		Np = ids.shape[0]
	return Np


def bndTPs(fname):
	"""
	Get the number of particles and the minimum/maximum particle IDs from the H5Part file.

	Args:
		fname (str): The path to the H5Part file.

	Returns:
		tuple: A tuple containing the number of particles (Np), the minimum particle ID (nS), and the maximum particle ID (nE).
	"""
	with h5py.File(fname, 'r') as hf:
		grp = hf.get("Step#0")
		ids = grp.get("id")[()]
		Np = ids.shape[0]
		nS = ids.min()
		nE = ids.max()
	return Np, nS, nE


#Find array index for a given particle ID (ie if block doesn't start at 1)
def locPID(fname, pid):
	"""
	Find the array index of a particle with a given ID in the H5Part file.

	Args:
		fname (str): The path to the H5Part file.
		pid (int): The ID of the particle to find.

	Returns:
		int or None: The array index of the specified particle or None if the given particle ID is not found in the H5Part file.
	"""
	with h5py.File(fname, 'r') as hf:
		grp = hf.get("Step#0")
		ids = grp.get("id")[()]
		isP = (ids == pid)
		loc = isP.argmax()
		if (ids[loc] != pid):
			print("Didn't find particle %d ..."%(pid))
			loc = None
			quit()
	return loc


#Given an h5part file, create a time series for a single particle w/ ID = pid
def getH5pid(fname, vId, pid):
	"""
	Retrieve a time series of a specified variable for a given particle from an H5Part file.

	Args:
		fname (str): The path to the H5Part file.
		vId (str): The variable ID to create a time series of.
		pid (int): The particle ID to retrieve.

	Returns:
		tuple: A tuple containing two arrays - the time array and the time series of the specified variable.
	"""
	p0 = locPID(fname, pid)  # Find particle in array
	Nt,sIds = kH5.cntSteps(fname)  # Find number of slices

	V = np.zeros(Nt)
	t = np.zeros(Nt)
	with h5py.File(fname, 'r') as hf:
		for n in range(Nt):
			# Create gId
			gId = "Step#%d"%(n)
			grp = hf.get(gId)
			t[n] = grp.attrs.get("time")
			V[n] = (grp.get(vId)[()])[p0]
	return t, V



#Given an h5part file, create a time series from an input string
def getH5p(fname, vId, Mask=None):
	"""
	Retrieve a time series of a specified variable for a subset of particles from an H5Part file.

	Args:
		fname (str): The file path of the H5Part file.
		vId (str): The variable ID to create a time series of.
		Mask (ndarray, optional): An optional mask to select a subset of particles to include.

	Returns:
		ndarray: An array of time values.
		ndarray: A 2D array of the time series of the selected variable. 
				 Dimensions: Nt x Np, where Nt is the length of the time series and Np is the number of particles that satify the Mask.

	If Mask is not provided, the function returns the time series for all particles within the H5Part file.
	If Mask is provided, the function returns the time series for particles with the applied mask.
	"""
	Nt,sIds = kH5.cntSteps(fname)
	Np = cntTPs(fname)
	t = np.zeros(Nt)
	V = np.zeros((Nt, Np))
	with h5py.File(fname, 'r') as hf:
		for n in range(Nt):
			# Create gId
			gId = "Step#%d"%(n)
			grp = hf.get(gId)
			t[n] = grp.attrs.get("time")
			V[n,:] = grp.get(vId)[()]
	if Mask is None:
		return t, V
	else:
		return t, V[:,Mask]


#Given an h5p file and step, get one slice of data
def getH5pT(fname, vID="isIn", nStp=0):
	"""
	Retrieve a specific variable from an H5Part file for all particles at a given time step.

	Args:
		fname (str): The path to the H5Part file.
		vID (str): The name of the variable to retrieve. Default is "isIn".
		nStp (int): The step number to pull the data from. Default is 0.

	Returns:
		1D array containing the variable a the given step. Dimensions: Np - number of particles in the H5Part file.
	"""
	with h5py.File(fname, 'r') as hf:
		gID = "Step#%d"%(nStp)
		V = hf.get(gID).get(vID)[()]

	return V