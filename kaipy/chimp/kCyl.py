#Various routines to deal with K-Cylinders from PSDs
import numpy as np
import datetime
import h5py
import kaipy.kaiViz as kv

#Get grid from K-Cyl
def getGrid(fIn, do4D=False):
	"""
	Retrieve grid data from an HDF5 file.

	Args:
		fIn (str): The path to the HDF5 file.
		do4D (bool, optional): Flag indicating whether to retrieve 4D data. Default is False.

	Returns:
		xx (ndarray): X-coordinate values of the grid.
		yy (ndarray): Y-coordinate values of the grid.
		Ki (ndarray): Original 3D data.
		Kc (ndarray): Averaged 3D data.
		Ai (ndarray, optional): Original 4D data. Only returned if do4D is True.
		Ac (ndarray, optional): Averaged 4D data. Only returned if do4D is True.
	"""
	with h5py.File(fIn, 'r') as hf:
		X3 = hf["X"][()].T
		Y3 = hf["Y"][()].T
		Z3 = hf["Z"][()].T
		if do4D:
			Ai = hf["A"][()].T

	xx = X3[:, :, 0]
	yy = Y3[:, :, 0]
	Zi = Z3[0, 0, :]
	Ki = 10 ** Zi
	Kc = 0.5 * (Ki[0:-1] + Ki[1:])
	if do4D:
		Ac = 0.5 * (Ai[0:-1] + Ai[1:])
		return xx, yy, Ki, Kc, Ai, Ac
	else:
		return xx, yy, Ki, Kc
	
	
def getSlc(fIn, nStp=0, vID="jPSD", doWrap=False):
	"""
	Retrieve a slice of data from an HDF5 file.

	Args:
		fIn (str): The path to the HDF5 file.
		nStp (int): The step number.
		vID (str): The variable ID.
		doWrap (bool): Whether to wrap the data.

	Returns:
		V (ndarray): The retrieved slice of data.
	"""
	gID = "Step#%d" % (nStp)
	with h5py.File(fIn, 'r') as hf:
		V = hf[gID][vID][()].T
	if (doWrap):
		return kv.reWrap(V)
	else:
		return V


#Pressure anisotropy
#doAsym : Px/Pz-1
#!doAsym: Px/(Px+Pz)

def PIso(fIn, nStp=0, pCut=1.0e-3, doAsym=False):
	"""
	Calculate the isotropy parameter for a given input file.

	Args:
		fIn (str): The input file path.
		nStp (int, optional): The number of steps. Default is 0.
		pCut (float, optional): The cutoff value. Default is 1.0e-3.
		doAsym (bool, optional): Flag to calculate asymmetric isotropy parameter. Default is False.

	Returns:
		pR (numpy.ndarray): The calculated isotropy parameter.

	"""
	Pxy = getSlc(fIn, nStp, "Pxy")
	Pz = getSlc(fIn, nStp, "Pz")
	Pk = 2 * Pxy + Pz
	Nx, Ny = Pz.shape
	pR = np.zeros((Nx, Ny))

	for i in range(Nx):
		for j in range(Ny):
			if Pk[i, j] > pCut and Pz[i, j] > pCut:
				if doAsym:
					pR[i, j] = Pxy[i, j] / Pz[i, j] - 1.0
				else:
					pR[i, j] = Pxy[i, j] / (Pxy[i, j] + Pz[i, j])
			else:
				if doAsym:
					pR[i, j] = 0.0
				else:
					pR[i, j] = np.nan
	return pR


#Equatorial grids (option for wrapping for contours)
def getEQGrid(fIn, doCenter=False, doWrap=False):
	"""
	Get the equidistant grid coordinates from an HDF5 file.

	Args:
		fIn (str): The path to the HDF5 file.
		doCenter (bool): Flag indicating whether to center the grid coordinates. Default is False.
		doWrap (bool): Flag indicating whether to wrap the grid coordinates. Default is False.

	Returns:
		If doCenter is False:
			xx (ndarray): The x-coordinates of the grid.
			yy (ndarray): The y-coordinates of the grid.
		If doCenter is True and doWrap is False:
			xxc (ndarray): The centered x-coordinates of the grid.
			yyc (ndarray): The centered y-coordinates of the grid.
		If doCenter and doWrap are both True:
			xxc (ndarray): The wrapped and centered x-coordinates of the grid.
			yyc (ndarray): The wrapped and centered y-coordinates of the grid.
	"""
	if doWrap:
		doCenter = True

	with h5py.File(fIn, 'r') as hf:
		xx = hf["X"][()].T
		yy = hf["Y"][()].T

	if not doCenter:
		return xx, yy

	Ngi, Ngj = xx.shape
	Ni = Ngi - 1
	Nj = Ngj - 1
	xxc = np.zeros((Ni, Nj))
	yyc = np.zeros((Ni, Nj))

	xxc = 0.25 * (xx[0:Ngi - 1, 0:Ngj - 1] + xx[1:Ngi, 0:Ngj - 1] + xx[0:Ngi - 1, 1:Ngj] + xx[1:Ngi, 1:Ngj])
	yyc = 0.25 * (yy[0:Ngi - 1, 0:Ngj - 1] + yy[1:Ngi, 0:Ngj - 1] + yy[0:Ngi - 1, 1:Ngj] + yy[1:Ngi, 1:Ngj])

	if not doWrap:
		return xxc, yyc
	else:
		return kv.reWrap(xxc), kv.reWrap(yyc)
	
	

