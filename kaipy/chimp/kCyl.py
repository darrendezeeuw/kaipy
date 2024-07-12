#Various routines to deal with K-Cylinders from PSDs
import numpy as np
import datetime
import h5py
import kaipy.kaiViz as kv

#Get grid from K-Cyl
def getGrid(fIn, do4D=False):
	"""
	Retrieve grid data (2D equatorial location, energy, pitch angle) from an HDF5 file containing PSD output.

	Args:
		fIn (str): The path to the PSD HDF5 file.
		do4D (bool, optional): Flag indicating whether to retrieve 4D grid (including pitch angle). Default is False.

	Returns:
		xx (ndarray): X-coordinates of the equatorail grid.
		yy (ndarray): Y-coordinates of the equatorail grid.
		Ki (ndarray): 1D array of energy grid at cell edges.
		Kc (ndarray): 1D array of energy grid at cell centers.
		Ai (ndarray, optional): 1D array of pitch angle grid at cell edges. Only returned if do4D is True.
		Ac (ndarray, optional): 1D array of pitch angle grid at cell centers. Only returned if do4D is True.
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
	Retrieve a specified variable at a given time step of data from an HDF5 file.

	Args:
		fIn (str): The path to the HDF5 file.
		nStp (int): The step number.
		vID (str): The variable ID.
		doWrap (bool)  : Flag indicating whether to add an extra layer in phi to the cell-centered grid coordinates. Useful for contour plots in polar coordinates. Default is False.

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
	Calculate the pressure anisotropy for a given input file.

	Args:
		fIn (str): The input file path.
		nStp (int, optional): The number of steps. Default is 0.
		pCut (float, optional): The minimum pressure value to consider. Default is 1.0e-3.
		doAsym (bool, optional): Flag to calculate Pxy/Pz-1, otherwise it will return Pxy/(Pxy+Pz). Default is False.

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
	Get the equatorial grid coordinates from an HDF5 file.

	Args:
		fIn (str): The path to the HDF5 file.
		doCenter (bool): Flag indicating whether to output the cell centered grid coordinates. Default is False.
		doWrap (bool)  : Flag indicating whether to add an extra layer in phi to the cell-centered grid coordinates. Useful for contour plots in polar coordinates. Default is False.

	Returns:
		If doCenter is False:
			xx (ndarray): The x-coordinates of the grid at the cell edges.
			yy (ndarray): The y-coordinates of the grid at the cell edges.
		If doCenter is True and doWrap is False:
			xxc (ndarray): The cell centered x-coordinates of the grid.
			yyc (ndarray): The cell centered y-coordinates of the grid.
		If doWrap is True:
			xxc (ndarray): The wrapped and cell centered x-coordinates of the grid.
			yyc (ndarray): The wrapped and cell centered y-coordinates of the grid.
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
	
	

