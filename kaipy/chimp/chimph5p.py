#Various routines to deal with h5p data
import numpy as np
import h5py

#Return time at step n
def tStep(fname, nStp=0):
	"""
	Get the time attribute for a given step in an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		nStp (int): The step number. Default is 0.

	Returns:
		float: The time attribute for the specified step.
	"""
	with h5py.File(fname, 'r') as hf:
		gID = "Step#%d" % (nStp)
		grp = hf.get(gID)
		t = grp.attrs.get("time")
	return t

	
#Count number of timesteps in an h5p file
def cntSteps(fname):
	"""
	Count the number of steps in an H5 file.

	Args:
		fname (str): The path to the H5 file.

	Returns:
		int: The number of steps in the H5 file.
	"""
	with h5py.File(fname, 'r') as hf:
		grps = hf.values()
		grpNames = [str(grp.name) for grp in grps]
		Steps = [stp for stp in grpNames if "/Step#" in stp]
		nSteps = len(Steps)

		return nSteps
	

#Count number of particles in an h5p file
def cntTPs(fname):
	"""
	Count the number of time points in the given H5 file.

	Args:
		fname (str): The path to the H5 file.

	Returns:
		int: The number of time points in the H5 file.
	"""
	with h5py.File(fname, 'r') as hf:
		grp = hf.get("Step#0")
		ids = grp.get("id")[()]
		Np = ids.shape[0]
	return Np


def bndTPs(fname):
	"""
	Get the number of particles, minimum ID, and maximum ID from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.

	Returns:
		tuple: A tuple containing the number of particles (Np), the minimum ID (nS), and the maximum ID (nE).
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
	Find the location of a particle with a given ID in an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		pid (int): The ID of the particle to locate.

	Returns:
		int or None: The index of the particle in the file, or None if the particle is not found.
	"""
	with h5py.File(fname, 'r') as hf:
		grp = hf.get("Step#0")
		ids = grp.get("id")[()]
		isP = (ids == pid)
		loc = isP.argmax()
		if (ids[loc] != pid):
			print("Didn't find particle %d ..." % (pid))
			loc = None
			quit()
	return loc


#Given an h5part file, create a time series for a single particle w/ ID = pid
def getH5pid(fname, vId, pid):
	"""
	Retrieve the time and values of a specific particle from an H5 file.

	Args:
		fname (str): The path to the H5 file.
		vId (str): The ID of the values to retrieve.
		pid (int): The ID of the particle to retrieve.

	Returns:
		tuple: A tuple containing two arrays - the time array and the values array.
	"""
	p0 = locPID(fname, pid)  # Find particle in array
	Nt = cntSteps(fname)  # Find number of slices

	V = np.zeros(Nt)
	t = np.zeros(Nt)
	with h5py.File(fname, 'r') as hf:
		for n in range(Nt):
			# Create gId
			gId = "Step#%d" % (n)
			grp = hf.get(gId)
			t[n] = grp.attrs.get("time")
			V[n] = (grp.get(vId)[()])[p0]
	return t, V



#Given an h5part file, create a time series from an input string
def getH5p(fname, vId, Mask=None):
	"""
	Retrieve time and voltage data from an H5p file.

	Args:
		fname (str): The file path of the H5p file.
		vId (str): The identifier of the voltage data to retrieve.
		Mask (ndarray, optional): An optional mask to apply to the voltage data.

	Returns:
		ndarray: An array of time values.
		ndarray: An array of voltage values.

	If Mask is not provided, the function returns the entire voltage data.
	If Mask is provided, the function returns the voltage data with the applied mask.
	"""
	Nt = cntSteps(fname)
	Np = cntTPs(fname)
	t = np.zeros(Nt)
	V = np.zeros((Nt, Np))
	with h5py.File(fname, 'r') as hf:
		for n in range(Nt):
			# Create gId
			gId = "Step#%d" % (n)
			grp = hf.get(gId)
			t[n] = grp.attrs.get("time")
			V[n, :] = grp.get(vId)[()]
	if Mask is None:
		return t, V
	else:
		return t, V[:, Mask]


#Given an h5p file and step, get one slice of data
def getH5pT(fname, vID="isIn", nStp=0, cutIn=False):
	"""
	Retrieve a specific dataset from an H5p file.

	Args:
		fname (str): The path to the H5p file.
		vID (str): The name of the dataset to retrieve. Default is "isIn".
		nStp (int): The step number. Default is 0.
		cutIn (bool): Whether to cut in the dataset. Default is False.

	Returns:
		The retrieved dataset.
	"""
	with h5py.File(fname, 'r') as hf:
		gID = "Step#%d" % (nStp)
		V = hf.get(gID).get(vID)[()]

	return V