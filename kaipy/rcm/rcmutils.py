import numpy as np
import h5py as h5

import kaipy.kdefs as kd
import kaipy.kaiH5 as kh5


#------
# Data containers
#------

class RCMSpeciesInfo(object):
	"""
	Container for RCM species

	Args:
		N      (int): Number of energy channels
		flav   (int): Species "flavor
					    e.g. 1 = electrons, 2 = protons
		kStart (int): Starting index in Nk-length arrays (e.g. eeta)
		kEnd   (int): Ending index +1 in Nk-length arrays
					    This way, array[kStart:kEnd] gives entire species
		alamc  List(float): List of grid-centered lambda values in units of [eV * (Rx/nT)**(2/3)]
	"""

	def __init__(self, N, flav, kStart, kEnd, alamc):
		self.N      = N
		self.flav   = flav
		self.kStart = kStart
		self.kEnd   = kEnd
		self.alamc  = alamc

class RCMInfo(kh5.H5Info):
	"""
	Extends H5Info to grab RCM-specific info

	Additional Args:
		config_fname (str): Full path to rcmconfig.h5 file. Used to get species information
	"""

	def __init__(self, h5fname, config_fname, noSubsec=True):
		super().__init__(h5fname, noSubsec)
		
		# Populate species info
		self.species = []
		with h5.File(config_fname) as f5:
			alamc = f5['alamc'][:]
			flavs = f5['ikflavc'][:]
			Nk = len(alamc)

			kPnt = 0
			# Check for plasmasphere
			if abs(alamc[0]) < 1e-12:
				self.species.append(
					RCMSpeciesInfo(
						1, flavs[0],  # N, flav
						0, 1,  # kStart, kEnd
						[0]  # alamc
  					)
				)
				kPnt += 1
			# Scrape remaining species
			doContinue = True
			while doContinue:
				# Each iteration collects information of the next species as defined in rcmconfig
				kStart = kPnt
				flav = flavs[kStart]
				while flavs[kPnt] == flav and kPnt < Nk-1: kPnt += 1  
					# First condition moves to start of next species, second condition and condition below handles last species in the list
				if kPnt == Nk-1: 
					kPnt += 1
					doContinue = False
				kEnd = kPnt
				self.species.append(
					RCMSpeciesInfo(
						(kEnd - kStart), flav,  #[N, flav
						kStart, kEnd,
						alamc[kStart:kEnd]
					)
				)
	def __test__(self, config_fname):
		"""
		Performs a test on an already-populated object to ensure its species info matches that in the provided file

		Args:
			config_fname (str): Full path to rcmconfig.h5

		Returns:
			result (bool): Whether test passes or not
		"""

		with h5.File(config_fname, 'r') as f5:
			alamc = f5['alamc'][:]
			flavs = f5['ikflavc'][:]
			testPass = True
			for spc in self.species:
				if not np.all(flavs[spc.kStart:spc.kEnd] == spc.flav):
					# If False, we are mixing species together
					testPass = False
				if (not np.all(alamc[spc.kStart:spc.kEnd] == spc.alamc)):
					testPass = False
				if spc.N != len(spc.alamc):
					testPass = False
			return testPass
#------
# Factors
#------
massi = kd.Mp_cgs*1e-3 # mass of ions in kg
masse = kd.Me_cgs*1e-3 # mass of lectrons in kg
nt    = 1.0e-9 # nt
ev    = kd.eCharge # 1ev in J
rx    = kd.Re_cgs*1e-2 # radius of earth in m

pressure_factor   = 2./3.*ev/rx*nt
specFlux_factor_i = 1/np.pi/np.sqrt(8)*np.sqrt(ev/massi)*nt/rx/1.0e1  # [units/cm^2/keV/str]
specFlux_factor_e = 1/np.pi/np.sqrt(8)*np.sqrt(ev/masse)*nt/rx/1.0e1  # [units/cm^2/keV/str]
TINY = 1.0e-8

def updateFactors(rxNew):
	"""
	Update the factors used in calculations based on the given value of rx.

	Parameters:
	rxNew (float): The new value of rx in meters.

	Returns:
	None
	"""

	global rx
	global pressure_factor
	global specFlux_factor_i
	global specFlux_factor_e

	rx = rxNew  # [m]

	pressure_factor   = 2./3.*ev/rx*nt
	specFlux_factor_i = 1/np.pi/np.sqrt(8)*np.sqrt(ev/massi)*nt/rx/1.0e1  # [units/cm^2/keV/str]
	specFlux_factor_e = 1/np.pi/np.sqrt(8)*np.sqrt(ev/masse)*nt/rx/1.0e1  # [units/cm^2/keV/str]

#------
# Lambda Channels
#------

#electrons: kStart = kStart, kEnd = kIon
#ions: kStart = kIon, kEnd = len(rcmS0['alamc'])
def getSpecieslambdata(rcmS0, species='ions'):
	"""
	Generates useful lambda information for a given species in the RCM model.

	Args:
		rcmS0 (h5py.Group): Step#X group from an rcm.h5 output file.
		species (str): Species for which to calculate lambdas. Default is 'ions'.

	Returns:
		dict: Dictionary containing the calculated lambdas and other information.
			- kStart (int): Start index of the lambdas.
			- kEnd (int): End index of the lambdas.
			- ilamc (ndarray): Cell centers.
			- ilami (ndarray): Cell interfaces.
			- lamscl (ndarray): Calculated lambdas.
	"""
	# Determine what our bounds are
	kStart = 1 if rcmS0['alamc'][0] == 0 else 0  # Check channel 0 for plasmasphere
	kIon = (rcmS0['alamc'][kStart:] > 0).argmax() + kStart
	if species == 'electrons':
		kEnd = kIon
	elif species == 'ions':
		kStart = kIon
		kEnd = len(rcmS0['alamc'])

	ilamc = rcmS0['alamc'][kStart:kEnd]  # Cell centers
	Nk = len(ilamc)
	ilami = np.zeros(Nk + 1)  # Cell interfaces
	for n in range(0, Nk - 1):
		ilami[n + 1] = 0.5 * (ilamc[n] + ilamc[n + 1])
	ilami[Nk] = ilamc[-1] + 0.5 * (ilamc[-1] - ilamc[-2])

	ilamc = np.abs(ilamc)
	ilami = np.abs(ilami)
	lamscl = np.diff(ilami) * np.sqrt(ilamc)

	result = {
		'kStart': kStart,
		'kEnd': kEnd,
		'ilamc': ilamc,
		'ilami': ilami,
		'lamscl': lamscl
	}
	return result

#------
# Domain
#------

def getClosedRegionMask(s5):
	"""
	Returns a boolean mask indicating closed regions based on the given input.

	Args:
		s5 (h5py.Group): An rcm.h5 Step#X group containing the required data for the calculation.

	Returns:
		bool: A boolean mask indicating closed regions.

	"""
	rmin = np.sqrt(s5['rcmxmin'][:]**2 + s5['rcmymin'][:]**2)

	mask = np.full(s5['rcmxmin'].shape, False)
	mask = (s5['rcmvm'][:] <= 0) | (rmin < 1.0)

	return mask

#------
# Cumulative pressure
#------
def getCumulPress(ilamc, vm, eetas, doFraction=False):
	"""
	Returns a 3D array of cumulative pressures.

	Args:
		ilamc (array-like): [Nk] lambda values (Nk NOT the full number of energy channels, can be subset)
		vm (array-like): [Nj,Ni] vm value
		eetas (array-like): [Nk,Nj,Ni] eetas corresponding to ilamc values
		doFraction (bool, optional): If True, returns cumulative pressure fraction instead of cumulative pressure. 
			Defaults to False.

	Returns:
		array-like: [Nk, Nj, Ni] array of cumulative pressures or cumulative pressure fractions.

	"""
	Nk = len(ilamc)
	Nj, Ni = vm.shape
	pCumul = np.zeros((Nk, Nj, Ni))

	ilam_kji = ilamc[:, np.newaxis, np.newaxis]
	vm_kji = vm[np.newaxis, :, :]

	pPar = pressure_factor * ilam_kji * eetas * vm_kji ** 2.5 * 1E9  # [Pa -> nPa], partial pressures for each channel

	pCumul[0] = pPar[0]  # First bin's partial press = cumulative pressure
	for k in range(1, Nk):
		pCumul[k] = pCumul[k - 1] + pPar[k]  # Get current cumulative pressure by adding this bin's partial onto last bin's cumulative

	if not doFraction:
		return pCumul
	else:
		return pCumul / pCumul[-1]


def getValAtLoc_linterp(val, xData, yData, getAxis='y'):
	"""
	Use linear interpolation to find the desired x/y values in data.

	Args:
		val (float): The x/y value to find the y/x location of.
		xData (list): 1D array of x-axis values.
		yData (list): 1D array of y-axis values.
		getAxis (str, optional): The desired axis. If 'y', assumes targets are x values, and vice versa. Defaults to 'y'.

	Returns:
		float: The interpolated location value.

	"""
	idx = 0
	if getAxis == 'y': # target is x-axis value, find its location in xData
		while xData[idx+1] < val:
			idx += 1
	elif getAxis == 'x': # target is y-axis value, find its location in yData
		while yData[idx+1] < val:
			idx += 1
	m = (yData[idx+1] - yData[idx]) / (xData[idx+1] - xData[idx])
	b = yData[idx] - m * xData[idx]
	if getAxis == 'y':
		loc = m * val + b
	elif getAxis == 'x':
		loc = (val - b) / m

	return loc





