import numpy as np
import datetime
import os
import glob
import sys
from astropy.time import Time

import kaipy.kdefs as kd
import kaipy.kaiH5 as kH5

#todo move to kaidefs
isotfmt = '%Y-%m-%dT%H:%M:%S.%f'

# `to_center1D` is a lambda function that takes a 1D array `A` and returns a new array 
# where each element is the average of its adjacent elements in the original array.
to_center1D = lambda A: 0.5*(A[:-1]+A[1:])

# `to_center2D` is a lambda function that takes a 2D array `A` and returns a new array 
# where each element is the average of its adjacent elements in the original array.
to_center2D = lambda A: 0.25*(A[:-1,:-1]+A[1:,:-1]+A[:-1,1:]+A[1:,1:])

# `to_center3D` is a lambda function that takes a 3D array `A` and returns a new array 
# where each element is the average of its adjacent elements in the original array.
to_center3D = lambda A: 0.125*(A[:-1,:-1,:-1]+A[-1:,-1:,1:]+A[-1:,1:,-1:]+A[-1:,1:,1:]
                +A[1:,:-1,:-1]+A[1:,-1:,1:]+A[1:,1:,-1:]+A[1:,1:,1:])


def L_to_bVol(L, bsurf_nT=kd.EarthM0g*kd.G2nT):
	"""
	Calculates the flux tube volume [Rp/nT] from the given L shell [Rp].

	Parameters:
		L (float): L shell [Rp].
		bsurf_nT (float, optional): Surface field [nT]. Default is Earth's surface field.

	Returns:
		float: Flux tube volume [Rp/nT].
	"""
	colat = np.arcsin(np.sqrt(1.0/L))

	cSum = 35*np.cos(colat) - 7*np.cos(3*colat) +(7./5.)*np.cos(5*colat) - (1./7.)*np.cos(7*colat)
	cSum /= 64.
	s8 = np.sin(colat)**8
	V = 2*cSum/s8/bsurf_nT
	return V  # [Rp/nT]

def MJD2UT(mjd):
	"""
	Convert Modified Julian Date (MJD) to Coordinated Universal Time (UTC).

	Parameters:
		mjd: A single value or a list of MJD values.

	Returns:
		If given a single value, returns a single datetime.datetime object representing the corresponding UTC time.
		If given a list of values, returns a list of datetime.datetime objects representing the corresponding UTC times.
	"""

	UT = Time(mjd, format='mjd').isot
	if type(UT) == str:
		return datetime.datetime.strptime(UT, isotfmt)
	else:
		return [datetime.datetime.strptime(UT[n], isotfmt) for n in range(len(UT))]

def utIdx(utList,ut):
	""" 
	Finds the index of the closest datetime within an array of datetimes

	Parameters:
		utList (List[datetime.datetime]): List of datetimes to find index within
		ut (datetime.datetime): datetime to find closest index to

	Returns:
		Single integer value of the closest index within 'utList' to 'ut'
	"""
	return np.array([np.abs((utList[n]-ut).total_seconds()) for n in range(len(utList))]).argmin()

def pntIdx_2D(X2D, Y2D, pnt):
	"""
	Finds i, j, index within 'X2D' and 'Y2D' closest to 'pnt'
	Assumes cartesian coordinate system
		
	Parameters:
		X2D (ndarray[float]): 2D array of X values
		Y2D (ndarray[float]): 2D array of Y values
		pnt (List[float])   : [x, y] point to find closest index to

	Returns:
		Tuple of closest i, j index to point 'pnt'
	"""
	distSq = (X2D-pnt[0])**2 + (Y2D-pnt[1])**2  # Each source point's euclidian distance from 'pnt'
	aMinFlattened = distSq.argmin()  # Index of min distance, if it was a flattened array
	i, j = np.unravel_index(aMinFlattened, X2D.shape)  # Convert back into i,j location
	return i, j

def getRunInfo(fdir, ftag):
	"""
	Get information about the run.

	Args:
		fdir (str): The directory where the files are located.
		ftag (str): The file tag.

	Returns:
		tuple: A tuple containing the file name, isMPI flag, Ri, Rj, and Rk values.

	Raises:
		ValueError: If more than one parallel file is found or no MPI database is found.
	"""
	idStr = "_0000_0000_0000.gam.h5"
	isMPI = False
	Ri = 0
	Rj = 0
	Rk = 0
	fOld = os.path.join(fdir, ftag + '.h5')
	fNew = os.path.join(fdir, ftag + '.gam.h5')
	try:
		if os.path.exists(fOld):
			return fOld, isMPI, Ri, Rj, Rk
		if os.path.exists(fNew):
			return fNew, isMPI, Ri, Rj, Rk
		fIns = glob.glob(os.path.join(fdir, ftag) + '_*' + idStr)
		if len(fIns) > 1:
			raise ValueError('Should not find more than one parallel file')
		if len(fIns) == 0:
			raise ValueError('No MPI database found')
		else:
			isMPI = True
			fName = fIns[0]
			Ns = [int(s) for s in fName.split('_') if s.isdigit()]
			Ri = Ns[-5]
			Rj = Ns[-4]
			Rk = Ns[-3]
			return fName, isMPI, Ri, Rj, Rk
	except ValueError as ve:
		print(ve)
		sys.exit()

#transform from cartesian to spherical
def xyz2rtp(phi, theta, Ax, Ay, Az):
	"""
	Convert Cartesian coordinates (Ax, Ay, Az) to spherical coordinates (r, theta, phi).

	Parameters:
		phi (float): The azimuthal angle in radians.
		theta (float): The polar angle in radians.
		Ax (float): The x-coordinate in Cartesian coordinates.
		Ay (float): The y-coordinate in Cartesian coordinates.
		Az (float): The z-coordinate in Cartesian coordinates.

	Returns:
		Ar (float): The radial component in spherical coordinates.
		At (float): The azimuthal component in spherical coordinates.
		Ap (float): The polar component in spherical coordinates.
	"""
	Ar = Ax * np.cos(phi) * np.sin(theta) + Ay * np.sin(phi) * np.sin(theta) + Az * np.cos(theta)
	Ap = -Ax * np.sin(phi) + Ay * np.cos(phi)
	At = Ax * np.cos(phi) * np.cos(theta) + Ay * np.sin(phi) * np.cos(theta) - Az * np.sin(theta)
	return Ar, At, Ap

def rtp2rt(r, theta, phi):
	"""
	Convert spherical coordinates (r, theta, phi) to 2D polar (r, phi)

	Parameters:
		r     (float): radial dimension
		theta (float): [rad] meridional/zenith direction
		phi   (float): [rad] azimuthal direciton

	Returns:
		r     (float): radial dimension
		theta (float): [rad]azimuthal dimenison
	"""
	
	Ar = r*np.sin(theta)
	Atheta = phi
	return Ar, Atheta

# Use the Burton 1975 Formula to compute Dst from solar wind parameters
def burtonDst(secs, n, vx, vy, vz, bx, by, bz):
	"""
	Given time in seconds, density in cm^-3, velocity in km/s and magnetic field in nT,
	this function will return Dst computed from the Burton 1975 
	(doi: 10.1029/ja080i031p04204) model.

	Parameters:
		secs (array-like): Time in seconds.
		n (array-like): Density in cm^-3.
		vx (array-like): Velocity in km/s (x-component).
		vy (array-like): Velocity in km/s (y-component).
		vz (array-like): Velocity in km/s (z-component).
		bx (array-like): Magnetic field in nT (x-component).
		by (array-like): Magnetic field in nT (y-component).
		bz (array-like): Magnetic field in nT (z-component).

	Returns:
		dst (array-like): Dst values computed from the Burton 1975 model.

	Note:
		The input arrays should have the same length.

	References:
		Burton, R. K., et al. (1975). "The Dst index: A measure of geomagnetic activity." Journal of Geophysical Research, 80(31), 4204-4214. doi: 10.1029/ja080i031p04204
	"""
	v = np.sqrt(vx**2 + vy**2 + vz**2)
	b = np.sqrt(bx**2 + by**2 + bz**2)
	ey = 1.0e-3 * vx * bz
	a = 3.6e-5
	b = 0.2
	c = 20.0
	d = -1.5e-3
	dp = -1.2e-3
	fe = d * (np.maximum(0, ey - 0.5))
	pdyn = 1.6726e-27 * 1e6 * 6.248e18 * n * v**2
	sqrtpdyn = np.sqrt(pdyn)
	dst = 0 * pdyn
	for i in np.arange(len(dst) - 1) + 1:
		dst[i] = dst[i - 1] + (secs[i] - secs[i - 1]) * (fe[i] - a * (dst[i - 1] - b * sqrtpdyn[i] + c))
	return dst

def newellkp(secs, n, vx, vy, vz, bx, by, bz):
		"""
		Given time in seconds, density in cm^-3, velocity in km/s, and magnetic field in nT,
		this function will return Kp computed from the Newell 2008 (doi: 10.1029/2007ja012825).

		Parameters:
			secs (array-like): Time in seconds.
			n (array-like): Density in cm^-3.
			vx (array-like): Velocity in km/s (x-component).
			vy (array-like): Velocity in km/s (y-component).
			vz (array-like): Velocity in km/s (z-component).
			bx (array-like): Magnetic field in nT (x-component).
			by (array-like): Magnetic field in nT (y-component).
			bz (array-like): Magnetic field in nT (z-component).

		Returns:
			conextendkp (array-like): Kp values computed from the Newell 2008 model.

		Notes:
			Kp is a three-hour index, so the function computes a three-hour rolling average using convolution
			with the first and last value extended.
		"""
		v = np.sqrt(vx**2 + vy**2 + vz**2)
		newellCoup = newellcoupling(vx, vy, vz, bx, by, bz)
		kp = np.clip(0.05 + 2.244e-4 * newellCoup + 2.844e-6 * np.sqrt(n) * v**2, 0, 9.0)
		window = int(3 * 60 * 60 / (secs[1] - secs[0]))
		halfwindow = int(window / 2)
		begin = np.ones(halfwindow)
		end = np.ones(halfwindow)
		begin[:] = kp[0]
		end[:] = kp[-1]
		extendkp = np.concatenate((begin, kp, end))
		conextendkp = np.convolve(extendkp, np.ones(window) / window, mode='same')
		return conextendkp[halfwindow:-halfwindow]

def newellcoupling(vx, vy, vz, bx, by, bz):
	"""
	Given density in cm^-3, velocity in km/s, and magnetic field in nT,
	this function returns the Universal Coupling Function computed from the 
	Newell 2007 (doi: 10.1029/2006ja012015).

	Parameters:
		vx (float): X-component of velocity in km/s.
		vy (float): Y-component of velocity in km/s.
		vz (float): Z-component of velocity in km/s.
		bx (float): X-component of magnetic field in nT.
		by (float): Y-component of magnetic field in nT.
		bz (float): Z-component of magnetic field in nT.

	Returns:
		newcoup (float): The computed Universal Coupling Function.
	"""
	v = np.sqrt(vx**2 + vy**2 + vz**2)
	b = np.sqrt(bx**2 + by**2 + bz**2)
	thetac = np.abs(np.arctan2(by, bz))
	newcoup = np.power(v, 4.0/3.0) * np.power(b, 2.0/3.0) * np.power(np.sin(thetac/2.0), 8.0/3.0)
	return newcoup

#Read SymH from bcwind file
def GetSymH(fBC):
	"""
	Retrieve symh data from an HDF5 file.

	Parameters:
		fBC (str): The path to the HDF5 file.

	Returns:
		utData (numpy.ndarray): Array of UT data.
		tData (numpy.ndarray): Array of T data.
		dstData (numpy.ndarray): Array of symh data.
	"""
	mjdData = kH5.PullVar(fBC, 'MJD')
	tData = kH5.PullVar(fBC, 'T')
	dstData = kH5.PullVar(fBC, 'symh')
	utData = MJD2UT(mjdData)
	return utData, tData, dstData

def interpTSC(gridX, gridY, x, y, var):
	"""
	Interpolates a variable using the Triangular Shaped Cloud (TSC) method.

	Args:
		gridX (list): 3-element x grid values (center value is closest grid point to desired interpolation point)
		gridY (list): 3-element y grid values (center value is closest grid point to desired interpolation point)
		x (float): x-coordinate of the point of interest
		y (float): y-coordinate of the point of interest
		var (list): 3x3 values of the desired variable

	Returns:
		float: The interpolated value of the variable at the given point of interest.
	"""
	weights = interpTSCWeights(gridX, gridY, x, y)

	result = 0
	for i in range(3):
		for j in range(3):
			result += weights[i,j] * var[i,j]

	return result
def interpTSC(gridX, gridY, x, y, var):
	""" Conducts a Triangular Shaped Cloud (TSC) interpolation of a 2D variable.
	
	Paramters:
		gridX/gridY: 3-element x & y grid vals (center value is closest grid point to desired interpolation point)
		x: dim1 of point of interest
		y: dim2 of point of interest
		var: 3x3 values of desired variable
	"""

	weights = interpTSCWeights(gridX, gridY, x, y)

	result = 0
	for i in range(3):
		for j in range(3):
			result += weights[i,j]*var[i,j]

	return result

def interpTSCWeights(gridX, gridY, x, y):
	"""
	Interpolates the weights for a 2D point of interest using a 3x3 grid.

	Args:
		gridX (list): A list of 3 x-coordinates of the grid points.
		gridY (list): A list of 3 y-coordinates of the grid points.
		x (float): The x-coordinate of the point of interest.
		y (float): The y-coordinate of the point of interest.

	Returns:
		numpy.ndarray: A 3x3 array of weights for the point of interest.

	"""
	dx = np.abs(gridX[0]-gridX[1])
	dy = np.abs(gridY[0]-gridY[1])

	eta  = (x - gridX[1])/dx
	zeta = (y - gridY[1])/dy

	def weight1D(eta):
		return np.array([
				0.5*(0.5-eta)**2,
				0.75 - eta**2,
				0.5*(0.5+eta)**2
			])

	wX = weight1D(eta)
	wY = weight1D(zeta)

	w2D = np.zeros((3,3))
	for i in range(3):
		for j in range(3):
			w2D[i,j] = wX[i]*wY[j]
	return w2D


def dipoleShift(xyz, r):
    #Find L of this point
	L = dipoleL(xyz)

	# Avoid bad values if L<r, push as far as possible
	L = max(L,r)

	# Use r = L*cos^2(latitude)
	mlat = np.abs(np.acos(np.sqrt(r/L)))
	mlon = np.atan2(xyz[1],xyz[0])  # No change in longitude
	if (mlon<0): mlon = mlon+2*np.pi

	if (xyz[2]<0):
		mlat = -np.abs(mlat)
	
	# Get cartesian coordinates
	xyz_out = [r*np.cos(mlat)*np.cos(mlon),
			   r*np.cos(mlat)*np.sin(mlon),
			   r*np.sin(mlat)]
	return xyz_out


def dipoleL(xyz):
	
	z = xyz[2]

	rad = np.linalg.norm(xyz)
	lat = abs( np.arcsin(z/rad) )
	Leq = rad/( np.cos(lat)*np.cos(lat) )
	return Leq