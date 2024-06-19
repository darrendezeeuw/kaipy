import h5py as h5
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from astropy.time import Time
import datetime

import kaipy.kaiH5 as kh5
import kaipy.kaiViz as kv
import kaipy.kaijson as kj
import kaipy.kaiTools as kT
import kaipy.gamera.gampp as gampp
import kaipy.gamera.rcmpp as rcmpp

import kaipy.satcomp.scutils as scutils


#Optionals
doProgressBar = True
if doProgressBar:
	import progressbar

#Constants
massi = 1.67e-27 # mass of ions in kg
masse = 9.11e-31 # mass of lectrons in kg
ev = 1.607e-19 # 1ev in J
nt = 1.0e-9 # nt
re = 6.380e6 # radius of earth in m

pressure_factor   = 2./3.*ev/re*nt
specFlux_factor_i = 1/np.pi/np.sqrt(8)*np.sqrt(ev/massi)*nt/re/1.0e1  # [units/cm^2/keV/str]
specFlux_factor_e = 1/np.pi/np.sqrt(8)*np.sqrt(ev/masse)*nt/re/1.0e1  # [units/cm^2/keV/str]
TINY = 1.0e-8

to_center = lambda A: 0.25*(A[:-1,:-1]+A[1:,:-1]+A[:-1,1:]+A[1:,1:])

#Settings that should maybe be elsewhere
tkl_lInner = 2  # Exclude points within 2 Re

#Hard-coded filenames
RCM_TIME_JFNAME = 'rcm_times.json'
RCM_TKL_JFNAME = 'rcm_tkl.json'
RCM_WEDGE_JFNAME = 'rcm_wedgetkl.json'
RCM_EQLATLON_JFNAME = 'rcm_eqlatlon.json'
RCM_CUMULFRAC_JFNAME = 'rcm_cumulFrac.json'
MHDRCM_TIME_JFNAME = 'mhdrcm_times.json'

#Spacecraft strings for cdaweb retrieval
#Probably shouldn't be hard-coded
supportedSats = ["RBSPA", "RBSPB"]
supportedDsets = ["Hydrogen_omniflux_RBSPICE", 'Hydrogen_omniflux_RBSPICE_TOFXPHHHELT', "Electron_omniflux_RBSPICE", "Hydrogen_PAFlux_HOPE"]

#======
#Helpers
#======

#Generate json filename for given spacecraft dataset
def genSCD_jfname(jdir, scName, dSetName):
	"""
	Generates the JSON file name for a given satellite name and data set name.

	Args:
		jdir (str): The directory where the JSON file will be located.
		scName (str): The name of the satellite.
		dSetName (str): The name of the data set.

	Returns:
		str: The generated JSON file name.

	"""
	jfname = scName + "_" + dSetName + ".json"
	return os.path.join(jdir, jfname)


def genRCMTrack_jfname(jdir, scName):
	"""
	Generate the JSON file name for RCM track.

	Args:
		jdir (str): The directory where the JSON file will be located.
		scName (str): The name of the spacecraft.

	Returns:
		str: The full path of the JSON file name for RCM track.
	"""
	jfname = "rcmTrack_" + scName + ".json"
	return os.path.join(jdir, jfname)

#Return a 3D cube of a given variable to use in interpolation
def getVarCube(rcm5, varName, stepLow, stepHigh, ilon, ilat, k=None):
	def getVarCube(rcm5, varName, stepLow, stepHigh, ilon, ilat, k=None):
		"""
		Returns a variable cube from the given RCM5 dataset.

		Args:
			rcm5 (numpy.ndarray): The RCM5 dataset.
			varName (str): The name of the variable.
			stepLow (int): The lower time step.
			stepHigh (int): The higher time step.
			ilon (int): The longitude index.
			ilat (int): The latitude index.
			k (int, optional): The vertical level index. Defaults to None.

		Returns:
			numpy.ndarray: The variable cube.

		"""
	if k is None:
		v1 = np.expand_dims(rcm5[stepLow][varName][ilon:ilon+2, ilat:ilat+2], axis=2)
		v2 = np.expand_dims(rcm5[stepHigh][varName][ilon:ilon+2, ilat:ilat+2], axis=2)
	else:
		v1 = np.expand_dims(rcm5[stepLow][varName][k, ilon:ilon+2, ilat:ilat+2], axis=2)
		v2 = np.expand_dims(rcm5[stepHigh][varName][k, ilon:ilon+2, ilat:ilat+2], axis=2)
	return np.append(v1, v2, axis=2)

#electrons: kStart = kStart, kEnd = kIon
#ions: kStart = kIon, kEnd = len(rcmS0['alamc'])
def getSpecieslambdata(rcmS0, species='ions'):
	"""
	Calculate lambdata for a given species in the RCM model.

	Args:
		rcmS0 (dict): Dictionary containing RCM data.
		species (str): Species for which lambdata is calculated. Can be 'ions' or 'electrons'. Default is 'ions'.

	Returns:
		dict: Dictionary containing the calculated lambdata.

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

# TODO Ask Anthony why not just use the one in kaipy.kaiViz
def genVarNorm(var, doLog=False):
	"""
	Generate normalized colorbar .

	Parameters:
	- var: numpy array
		The variable for which to generate normalized values.
	- doLog: bool, optional
		Flag indicating whether to use logarithmic normalization. Default is False.

	Returns:
	- norm: normalized colorbar object.

	"""
	vMin = np.min(var[var>0])
	vMax = np.max(var)
	norm = kv.genNorm(vMin, vMax, doLog=doLog)

	return norm

# TODO Ask Anthony why not just use the one in kaipy.kaiViz
def get_aspect(ax):
	"""
	Calculate the aspect ratio of an axis.

	Args:
		ax: The axis object for which to calculate the aspect ratio.

	Returns:
		The aspect ratio of the axis.

	"""
	from operator import sub

	# Total figure size
	figW, figH = ax.get_figure().get_size_inches()

	# Axis size on figure
	_, _, w, h = ax.get_position().bounds

	# Ratio of display units
	disp_ratio = (figH * h) / (figW * w)

	# Ratio of data units
	# Negative over negative because of the order of subtraction
	data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

	return disp_ratio #/ data_ratio

#======
#Main work
#======

#Given sc and dataset name (according to above dict), grab specifically omnidirecitonal differential flux
def getSCOmniDiffFlux(scName, dSetName, t0, t1, jdir=None, forceCalc=False):
	"""
	Retrieves spacecraft data and calculates omnidirectional differential flux.

	Args:
		scName (str): Name of the spacecraft.
		dSetName (str): Name of the dataset.
		t0 (float): Start time of the data retrieval.
		t1 (float): End time of the data retrieval.
		jdir (str, optional): Directory to store the JSON file. Defaults to None.
		forceCalc (bool, optional): Flag to force calculation even if JSON file exists. Defaults to False.

	Returns:
		ephdata (dict): Ephemeris data.
		dataset (dict): Dataset containing the calculated omnidirectional differential flux.

	Raises:
		None

	Examples:
		# Retrieve spacecraft data and calculate omnidirectional differential flux
		ephdata, dataset = getSCOmniDiffFlux("Satellite1", "Dataset1", 0, 10)

	"""
	if jdir is not None:
		dojson = True
		jfname = genSCD_jfname(jdir, scName, dSetName)
	else:
		dojson = False

	if dojson and not forceCalc:
		if os.path.exists(jfname):
			print("Grabbing spacecraft data from " + jfname)
			data = kj.load(jfname)
			ephdata = data['ephdata']
			dataset = data['dataset']
			return ephdata, dataset

	print("Pulling spacecraft data from cdaweb")
	# TODO: Add all desired datasets

	scStrs = scutils.getScIds()
	satStrs = scStrs[scName]
	ephemStrs = satStrs['Ephem']
	dsetStrs = satStrs[dSetName]
	epochStr = "Epoch" if "EpochStr" not in dsetStrs.keys() else dsetStrs['EpochStr']
	energyStr = dsetStrs['EnergyStr']

	# First get ephem data
	s, ephdata = scutils.pullVar(ephemStrs['Id'], ephemStrs['Data'], t0, t1)
	s, data = scutils.pullVar(dsetStrs['Id'], dsetStrs['Data'], t0, t1, doVerbose=True)

	dataset = {}
	dataset['name'] = scName
	species = 'electrons' if 'E' == dSetName[0] else 'ions'
	dataset['species'] = species
	dataset['epoch'] = data[epochStr]
	
	# Turn each dataset's data into omniflux
	if dSetName in ['Hydrogen_omniflux_RBSPICE', 'Hydrogen_omniflux_RBSPICE_TOFXPHHHELT', 'Electron_omniflux_RBSPICE']:
		# Already got omni flux, no problem
		ofStr = dsetStrs['OmnifluxStr']
		if dSetName == 'Hydrogen_omniflux_RBSPICE':
			dataset['OmniDiffFlux'] = data[ofStr]*1E-3  # Diferential flux [1/(MeV-cm^2-s-sr]*[1/MeV -> 1/keV]
			dataset['energies'] = data[energyStr]*1E3  # [MeV] -> [keV]
		else:
			dataset['OmniDiffFlux'] = data[ofStr]*1E-3  # Diferential flux [1/(MeV-cm^2-s-sr]*[1/MeV -> 1/keV]
			dataset['energies'] = data[energyStr]*1E3  # [MeV] -> [keV]
		
	elif dSetName == "Hydrogen_PAFlux_HOPE":
		# !!TODO: Properly calc OmniDiffFlux from FPDU. Currently just using pitch angle
		fluxStr = dsetStrs['PAFluxStr']
		iPA = np.abs(data['PITCH_ANGLE'] - 90).argmin()
		dataset['OmniDiffFlux'] = data[fluxStr][:,iPA,:]  # (epoch, energy)
		dataset['energies'] = data[energyStr]*1E-3  # [eV -> keV]

	# Pause to save to json
	if dojson:
		print("Saving to file")
		jdata = {'ephdata' : ephdata, 'dataset' : dataset}
		kj.dump(jfname, jdata)

	return ephdata, dataset

def getRCMtimes(rcmf5, mhdrcmf5, jdir=None, forceCalc=False):
	"""
	Grab RCM times, sIDs, and MJDs

	Args:
		rcmf5 (str): Path to the RCM hdf5 file.
		mhdrcmf5 (str): Path to the MHDRCM hdf5 file.
		jdir (str, optional): Directory to search for JSON files. Defaults to None.
		forceCalc (bool, optional): Flag to force calculation even if JSON files exist. Defaults to False.

	Returns:
		dict: Dictionary containing RCM time data.

	Notes:
		- If jdir is given, the function will try to find the files there. If not found, it will pull the data from the hdf5 files.
		- If JSON files exist and forceCalc is False, the function will load the data from the JSON files.
		- If JSON files do not exist or forceCalc is True, the function will calculate the RCM times based on the MHDRCM file.

	"""
	if jdir is not None:
		dojson = True
		rcmjfname = os.path.join(jdir, RCM_TIME_JFNAME)
		mhdrcmjfname = os.path.join(jdir, MHDRCM_TIME_JFNAME)
	else:
		dojson = False

	if dojson and not forceCalc:
		if os.path.exists(rcmjfname):
			print("Grabbing RCM time data from " + rcmjfname)
			# That's all that's needed, done
			return kj.load(rcmjfname)
		else:
			# Must create RCM MJD's based on MHDRCM file, so get MHDRCM data
			if os.path.exists(mhdrcmjfname):
				print("Grabbing MHDRCM time data from " + mhdrcmjfname)
				mhdrcmTimes = kj.load(mhdrcmjfname)
				mhdrcmT = mhdrcmTimes['T']
				mhdrcmMJDs = mhdrcmTimes['MJD']
			else:
				print("No usable files in " + jdir + ", grabbing all from hdf5's")

	if 'mhdrcmTimes' not in locals():
		print("Grabbing MHDRCM time data from " + mhdrcmf5)
		Nt, sIDs = kh5.cntSteps(mhdrcmf5)
		sIDs = np.sort(sIDs)
		mhdrcmT = kh5.getTs(mhdrcmf5, sIDs, aID='time')
		mhdrcmMJDs = kh5.getTs(mhdrcmf5, sIDs, aID='MJD')

		mhdrcmTimes = {'Nt': Nt,
					   'sIDs': sIDs,
					   'T': mhdrcmT,
					   'MJD': mhdrcmMJDs}
		# Pause to save to file
		if dojson:
			kj.dump(mhdrcmjfname, mhdrcmTimes)

	print("Grabbing RCM time data from " + rcmf5)
	Nt, sIDs = kh5.cntSteps(rcmf5)
	sIDs = np.sort(sIDs)
	sIDstrs = np.array(['Step#' + str(s) for s in sIDs])
	rcmT = kh5.getTs(rcmf5, sIDs, aID="time")

	if (mhdrcmT[:] == rcmT[:]).all():
		rcmMJDs = mhdrcmMJDs
	else:
		# !! This only works if all rcm steps are also in mhdrcm
		#    In the future: As long as a single timestep is in both, that MJD can be used to get all MJDs for rcm steps
		rcmMJDs = np.zeros((len(rcmT)))
		for i in range(len(rcmT)):
			idx = np.where(mhdrcmT == rcmT[i])[0]
			rcmMJDs[i] = mhdrcmMJDs[idx]

	rcmTimes = {'rcmf5': rcmf5,
				'mhdrcmf5': mhdrcmf5,
				'Nt': Nt,
				'sIDs': sIDs,
				'sIDstrs': sIDstrs,
				'T': rcmT,
				'MJD': rcmMJDs}
	# Pause to save rcm jfile
	if dojson:
		kj.dump(rcmjfname, rcmTimes)

	return rcmTimes

def getRCM_scTrack(trackf5, rcmf5, rcmTimes, jdir=None, forceCalc=False, scName=""):
	"""
	Pull RCM data along a given spacecraft track.

	Args:
		trackf5 (str): Path to the big spacecraft trajectory hdf5 file, generated from sctrack.x.
		rcmf5 (str): Path to the <tag>.rcm.h5 file.
		rcmTimes (dict): Dictionary containing RCM times information, including 'sIDstrs' and 'MJD'.
		jdir (str, optional): If included, will try to do json saving and loading to save time. Defaults to None.
		forceCalc (bool, optional): Whether to force calculation even if json file exists. Defaults to False.
		scName (str, optional): Name of the spacecraft. Defaults to "".

	Returns:
		dict: Dictionary containing RCM data along the spacecraft track, including time, MJD, MLAT, MLON, vm, energy, and eetas.

	"""
	# Function code goes here
	# ...


#TODO: Energy grid mapping in a nice, jsonizable way
#      Right now, just need to call this whenever you want it
def consolidateODFs(scData, rcmTrackData, eGrid=None, doPlot=False):
	"""
	Prepare the spacecraft and RCM track data for comparison.

	Args:
		scData (dict): Dictionary containing spacecraft data.
		rcmTrackData (dict): Dictionary containing RCM track data.
		eGrid (ndarray, optional): Energy grid to use. If not provided, fixed bins spanning the full energy range will be created.
		doPlot (bool, optional): Whether to plot the data. Default is False.

	Returns:
		result (dict): Dictionary containing the consolidated data.

	"""
	#scData determines which species we're using
	species = scData['species']
	rcmSpec = rcmTrackData[species]

	scEGrid = scData['energies']  # Might be 1D (fixed bins) or 2D (different energies over time)
	rcmEGrid = rcmSpec['energies']

	Nt_sc = len(scData['epoch'])
	#Manipulate sc EGrid, if 2D, so that time dimension is first index
	sce_shape = scEGrid.shape
	if len(sce_shape) > 1:
		timeAxis = -1
		
		for i in range(len(sce_shape)):
			if sce_shape[i] == Nt_sc:
				timeAxis = i
		if timeAxis == -1:
			print("Finding right time axis in sc data didn't work")
			return
		scEGrid = np.rollaxis(scEGrid, timeAxis, 0)
		
	#If given grid to use, do that
	#Else, make fixed bins spanning full energy range
	if eGrid is None:
		eMax = np.max([scEGrid.max(), rcmEGrid.max()])
		eMin = np.min([scEGrid[scEGrid>0].min(), rcmEGrid[rcmEGrid>0].min()])
		numPoints = 150
		eGrid = np.logspace(np.log10(eMin), np.log10(eMax), numPoints, endpoint=True)
	Ne = len(eGrid)
	e_lower = np.zeros(Ne)
	e_upper = np.zeros(Ne)
	for iEG in range(Ne):
		#e_lower[iEG] = eGrid[0] if iEG == 0 else (eGrid[iEG-1] + eGrid[iEG])/2
		#e_upper[iEG] = eGrid[-1] if iEG == Ne-1 else (eGrid[iEG+1] + eGrid[iEG])/2
		e_lower[iEG] = 2*eGrid[0]-eGrid[1] if iEG == 0 else (eGrid[iEG-1] + eGrid[iEG])/2
		e_upper[iEG] = 2*eGrid[-1]-eGrid[-2] if iEG == Ne-1 else (eGrid[iEG+1] + eGrid[iEG])/2
	
	#Map sc odf on given energy grid new new eGrid
	sc_odf = np.zeros((Nt_sc, Ne))
	for n in range(Nt_sc):
		
		#Might not need this check
		if len(sce_shape) > 1:
			#Assume scEGrid's are evenly spaced	
			scEG_interfaces = (scEGrid[n,1:]+scEGrid[n,:-1])/2
			scEG_lower = np.append(scEGrid[n,0], scEG_interfaces)
			scEG_upper = np.append(scEG_interfaces, scEGrid[n,-1])
			#scEG_lower = np.append(2*scEGrid[n,0]-scEGrid[n,1], scEG_interfaces)
			#scEG_upper = np.append(scEG_interfaces, 2*scEGrid[n,-1]-scEGrid[n,-2])
			mapWeights = scutils.getWeights_ConsArea(scEGrid[n], scEG_lower, scEG_upper, eGrid, e_lower, e_upper)
		else:
			scEG_interfaces = (scEGrid[1:]+scEGrid[:-1])/2
			scEG_lower = np.append(scEGrid[0], scEG_interfaces)
			scEG_upper = np.append(scEG_interfaces, scEGrid[-1])
			#scEG_lower = np.append(2*scEGrid[0]-scEGrid[1], scEG_interfaces)
			#scEG_upper = np.append(scEG_interfaces, 2*scEGrid[-1]-scEGrid[-2])
			mapWeights = scutils.getWeights_ConsArea(scEGrid, scEG_lower, scEG_upper, eGrid, e_lower, e_upper)

		for e in range(Ne):
			fillAmt = 0  # Fraction of given energy cell that's been filled with oldGrid stuff
			for ik in range(len(mapWeights[e])):
				k = mapWeights[e][ik][0]
				weight = mapWeights[e][ik][1]
				sc_odf[n,e] += weight*scData['OmniDiffFlux'][n,k]
				fillAmt += weight
			if fillAmt > 0.2 and fillAmt < 1:
				#print('n={},e={},sc_odf={},fillAmt={}'.format(n,e,sc_odf[n,e],fillAmt))
				sc_odf[n,e] /= fillAmt  # Fill cell
	
	Nt_rcm = len(rcmTrackData['T'])
	rcm_odf = np.zeros((Nt_rcm, len(eGrid)))
	#TODO: For RCM, we should re-map eetas and then recalc diffFlux
	for n in range(Nt_rcm):
		vm = rcmSpec['energies'][n,-1]/rcmSpec['ilamc'][-1]
		rcme_lower = rcmSpec['ilami'][:-1]*vm
		rcme_lower[0] = rcmSpec['energies'][n,0]  # Otherwise its zero, and it makes odd lines
		rcme_upper = rcmSpec['ilami'][1:]*vm
		mapWeights = scutils.getWeights_ConsArea(rcmSpec['energies'][n,:], rcme_lower, rcme_upper, eGrid, e_lower, e_upper)

		if doPlot:
			plt.scatter(eGrid, np.zeros(len(eGrid)), c="black")
			plt.scatter(e_lower, np.zeros(len(eGrid)), c="red")
			plt.scatter(rcmSpec['energies'][n], np.ones(len(rcme_lower)), c="black")
			plt.scatter(rcme_lower, np.ones(len(rcme_lower)), c="red")
			plt.xscale('log')
			plt.show()

		for e in range(Ne):
			for ik in range(len(mapWeights[e])):
				k = mapWeights[e][ik][0]
				weight = mapWeights[e][ik][1]
				rcm_odf[n,e] += weight*rcmSpec['diffFlux'][n,k]

	result = {}
	result['energyGrid'] = eGrid
	result['sc'] = {
		'name' : scData['name'],
		'time' : scData['epoch'],
		'diffFlux' : sc_odf}
	result['rcm'] = {
		'time' : rcmTrackData['MJD'],
		'origEGrid' : rcmEGrid,
		'origODF' : rcmSpec['diffFlux'],
		'diffFlux' : rcm_odf}

	return result

def getIntensitiesVsL(rcmf5, mhdrcmf5, sStart, sEnd, sStride, species='ions', eGrid=None, jdir=None, forceCalc=False):
	"""Calculate rcm intensities (summed diff flux).
	
	Args:
		rcmf5 (str): RCM HDF5 filename to pull xmin, ymin, zmin from.
		mhdrcmf5 (str): MHDRCM HDF5 filename to pull IOpen from.
		AxLvT (matplotlib.axes.Axes, optional): If given, will plot the resulting L shell vs. time intensity.
		jdir (str, optional): Give JSON directory to enable JSON usage (read/write results to file).
		forceCalc (bool, optional): If dataset already found in file, re-calculate anyways and overwrite it.
	
	Returns:
		result (dict): Dictionary containing the calculated intensities.
	
	"""

	if jdir is not None:
		dojson = True
		rcmjfname = os.path.join(jdir, RCM_TKL_JFNAME)
	else:
		dojson = False

	if dojson and not forceCalc:
		if os.path.exists(rcmjfname):
			print("Grabbing RCM tkl data from " + rcmjfname)
			return kj.load(rcmjfname)

	#Setup

	rcmTimes = getRCMtimes(rcmf5,mhdrcmf5,jdir=jdir)
	iTStart = np.abs(rcmTimes['sIDs']-sStart).argmin()
	iTEnd = np.abs(rcmTimes['sIDs']-sEnd).argmin()

	sIDstrs = rcmTimes['sIDstrs'][iTStart:iTEnd+1:sStride]
	nSteps = len(sIDstrs)
	rcm5 = h5.File(rcmf5,'r')
	rcmS0 = rcm5[sIDstrs[0]]
	mhdrcm5 = h5.File(mhdrcmf5,'r')

	if species == 'electrons':
		sf_factor = specFlux_factor_e
	elif species == 'ions':
		sf_factor = specFlux_factor_i
	alamData = getSpecieslambdata(rcmS0, species)

	alams_kxx = alamData['ilamc'][:, np.newaxis, np.newaxis]
	Nk = kEnd - kStart

	nlbins = 50
	lbins = np.linspace(2, 15, nlbins)

	if eGrid is None:
		eMin = 1E2  # [eV]
		eMax = 1E6  # [eV]
		Ne = 50
		eGrid = np.logspace(np.log10(eMin), np.log10(eMax), Ne, endpoint=True)
	Ne = len(eGrid)

	#Add endpoints to catch values outside of desired ranges
	lbins = np.concatenate(([lbins[0]-TINY], lbins, [lbins[-1]+TINY]))
	eGrid = np.concatenate(([eGrid[0]-TINY], eGrid, [eGrid[-1]+TINY]))

	rcmodf_tkl = np.zeros((nSteps, Ne+2, nlbins+2))
	rcmpress_tkl = np.zeros((nSteps, Ne+2, nlbins+2))

	if doProgressBar: bar = progressbar.ProgressBar(max_value=nSteps)
	for n in range(nSteps):
		if doProgressBar: bar.update(n)

		rcmS = rcm5[sIDstrs[n]]
		mhdrcmS = mhdrcm5[sIDstrs[n]]

		IOpen = mhdrcmS['IOpen']
		Ni, Nj = IOpen.shape  # Default should be (179, 90)
		#Shorten rcm data to match Ni, Nj of mhdrcm5
		vms = rcmS['rcmvm'][2:, :]
		xmins = rcmS['rcmxmin'][2:,:]
		ymins = rcmS['rcmymin'][2:,:]
		zmins = rcmS['rcmzmin'][2:,:]
		eetas = rcmS['rcmeeta'][kStart:kEnd,2:,:]

		vms_xij = vms[np.newaxis,:,:]

		#Calculate L shell for whole plane
		L_arr = scutils.xyz_to_L(xmins, ymins, zmins)  # [i,j]

		le_counts = np.zeros((Ne+2, nlbins+2))  # Keep track of how many entries into each bin so we can divide by it to get the average later

		energies = vms_xij * alams_kxx  # Should be [Nk, Ni, Nj]

		iL_arr = np.array([np.abs(lbins-i).argmin() for i in L_arr.flatten()]).reshape((Ni, Nj))
		iE_arr = np.array([np.abs(eGrid-e).argmin() for e in energies.flatten()]).reshape((Nk, Ni, Nj))

		diffFlux_Nk = sf_factor*energies*eetas/alamData['lamscl'][:, np.newaxis, np.newaxis]  # [k,i,j]

		pressure_kij = pressure_factor*alams_kxx*eetas*vms_xij**2.5 * 1E9  # [Pa -> nPa]

		for i in range(Ni):
			for j in range(Nj):
				for k in range(Nk):
					rcmodf_tkl[n, iE_arr[k,i,j], iL_arr[i,j]] += diffFlux_Nk[k,i,j]
					rcmpress_tkl[n, iE_arr[k,i,j], iL_arr[i,j]] += pressure_kij[k,i,j]
					le_counts[iE_arr[k,i,j], iL_arr[i,j]] += 1

		#Normalize to get avg. per count
		rcmodf_tkl[n,:,:] /= le_counts
		rcmpress_tkl[n,:,:] /= le_counts

	#Trim off the extra values
	lbins = lbins[1:-1]
	eGrid = eGrid[1:-1]
	rcmodf_tkl = rcmodf_tkl[:,1:-1,1:-1]
	rcmpress_tkl = rcmpress_tkl[:,1:-1,1:-1]
	
	# Sum across energy to get total avg. per L shell
	rcmpress_tl = np.ma.sum(np.ma.masked_invalid(rcmpress_tkl), axis=1)
	rcmodf_tl = np.ma.sum(np.ma.masked_invalid(rcmodf_tkl), axis=1)


	result = {}
	result['T']	         = rcmTimes['T'  ][iTStart:iTEnd+1:sStride]
	result['MJD']        = rcmTimes['MJD'][iTStart:iTEnd+1:sStride]
	result['lambda']     = alamData['ilamc']
	result['L_bins']     = lbins
	result['energyGrid'] = eGrid
	#result['nrg_tkl'] = rcmnrg_tkl
	result['odf_tkl']    = rcmodf_tkl
	result['odf_tl']    = rcmodf_tl
	result['press_tkl']  = rcmpress_tkl
	result['press_tl']  = rcmpress_tl

	if dojson: kj.dump(rcmjfname, result)

	return result

#TODO: Something odd with odf calculation
#TODO: Break out interpolator/mapper so they can be tested on their own and used by other functions
def getVarWedge(rcmf5, mhdrcmf5, sStart, sEnd, sStride, wedge_deg, species='ions', rcmTimes=None, eGrid=None, lGrid=None, jdir=None, forceCalc=False):
	"""Take a slice/wedge centered along the x axis (eq space), calculate average <var> vs. L and E.

	Args:
		rcmf5 (str): Filename of the RCM data in HDF5 format.
		mhdrcmf5 (str): Filename of the MHD RCM data in HDF5 format.
		sStart (float): Starting step value.
		sEnd (float): Ending step value.
		sStride (float): Step stride value.
		width_deg (float): Wedge width in degrees, centered around the x axis in the equatorial mapping of RCM data.
		rcmTimes (dict): Dictionary returned from the getRCMTimes() function.
		eGrid (ndarray, optional): Specific energy grid to map k energies to. Defaults to None.
		lbins (ndarray, optional): 1D array of L values to map to. Defaults to None.
		jdir (str, optional): Directory to find json files in. Defaults to None.

	Returns:
		dict: A dictionary containing the calculated results.

	Raises:
		None
	"""

	if jdir is not None:
		dojson = True
		rcmjfname = os.path.join(jdir, RCM_WEDGE_JFNAME)
	else:
		dojson = False

	if dojson and not forceCalc:
		if os.path.exists(rcmjfname):
			print("Grabbing RCM wedge-tkl data from " + rcmjfname)
			return kj.load(rcmjfname)

	#Setup
	if rcmTimes is None:
		rcmTimes = getRCMtimes(rcmf5,mhdrcmf5,jdir=jdir)
	iTStart = np.abs(rcmTimes['sIDs']-sStart).argmin()
	iTEnd = np.abs(rcmTimes['sIDs']-sEnd).argmin()

	sIDstrs = rcmTimes['sIDstrs'][iTStart:iTEnd+1:sStride]
	nSteps = len(sIDstrs)
	rcm5 = h5.File(rcmf5,'r')
	rcmS0 = rcm5[sIDstrs[0]]
	mhdrcm5 = h5.File(mhdrcmf5,'r')

	if species == 'electrons':
		sf_factor = specFlux_factor_e
	elif species == 'ions':
		sf_factor = specFlux_factor_i
	alamData = getSpecieslambdata(rcmS0, species)
	Nk = len(alamData['ilamc'])
	
	if lGrid is None:
		lMin = -10
		lMax = 8
		Nl = 52
		lGrid = np.linspace(lMin, lMax, Nl, endpoint=True)
	Nl = len(lGrid)
	
	if eGrid is None:
		eMin = 1E2  # [eV]
		eMax = 1E6  # [eV]
		Ne = 50
		eGrid = np.logspace(np.log10(eMin), np.log10(eMax), Ne, endpoint=True)
	Ne = len(eGrid)

	odf_tkl = np.zeros((nSteps, Ne, Nl))
	press_tkl = np.zeros((nSteps, Ne, Nl))

	if doProgressBar: bar = progressbar.ProgressBar(max_value=nSteps)
	for n in range(nSteps):
		if doProgressBar: bar.update(n)

		mhdrcmS = mhdrcm5[sIDstrs[n]]
		rcmS = rcm5[sIDstrs[n]]

		IOpen = mhdrcmS['IOpen']
		Ni, Nj = IOpen.shape  # Default should be (179, 90)
		#Shorten rcm data to match Ni, Nj of mhdrcm5
		vms = rcmS['rcmvm'][2:, :]
		xmins = rcmS['rcmxmin'][2:,:]
		ymins = rcmS['rcmymin'][2:,:]
		zmins = rcmS['rcmzmin'][2:,:]
		eetas = rcmS['rcmeeta'][alamData['kStart']:alamData['kEnd'],2:,:]

		L_arr = scutils.xyz_to_L(xmins, ymins, zmins) # [i,j]
		theta_arr = np.arctan2(ymins,xmins)*180/np.pi%360

		#Collect indicies of points within spatial bounds
		iPointCloud = []
		lPointCloud = []

		for i in range(Ni):
			for j in range(Nj):
				l_ij = L_arr[i,j]
				theta_ij = theta_arr[i,j]
				#Collect indicies of points within spatial bounds
				
				if IOpen[i,j] >= 0:
					continue

				if (theta_ij > 360-wedge_deg/2 or theta_ij < wedge_deg/2):  # dayside wedge
					if l_ij >= lGrid[0] and l_ij <= lGrid[-1] and abs(l_ij) > tkl_lInner:
						iPointCloud.append([i,j])
						lPointCloud.append(l_ij)	
				elif (theta_ij > 180-wedge_deg/2 and theta_ij < 180+wedge_deg/2): #nightside wedge
					if -l_ij >= lGrid[0] and -l_ij <= lGrid[-1] and abs(l_ij) > tkl_lInner:
						iPointCloud.append([i,j])
						lPointCloud.append(-l_ij)
	
		iPointCloud = np.array(iPointCloud)
		lPointCloud = np.array(lPointCloud)
		#Sort index array in order of ascending L values
		lSort = lPointCloud.argsort()
		iPointCloud = iPointCloud[lSort]
		lPointCloud = lPointCloud[lSort]
		Npc = len(iPointCloud)
		
		if Npc == 0:
			continue
		#Calc distance-based mapping weights for each lGrid index
		lMapWeights = [ [] for l in range(Nl)] # Nl x nx2 ,  nx[iPC, lWeight]
		wFlag = 9999
		for iLG in range(Nl):
			iLPC_lower = 0  # iPointCloud/lPointCloud index, 1 grid point below current lGrid index
			iLPC_upper = 0  # iPointCloud/lPointCloud index, 1 grid point above current lGrid index
			width_lower = abs(lGrid[iLG] - lGrid[iLG-1]) if iLG > 1 else wFlag
			width_upper = abs(lGrid[iLG+1] - lGrid[iLG]) if iLG < Nl-1 else wFlag
			if iLG != 0: 
				while iLPC_lower < Npc-1 and lPointCloud[iLPC_lower] < lGrid[iLG-1]:
					iLPC_lower += 1
			#Set upper iPLC index to next highest lbin value
			if iLG < Nl-1: 
				while iLPC_upper+1 < Npc and lPointCloud[iLPC_upper+1] < lGrid[iLG+1]:
					iLPC_upper += 1

			iLG_WM = []  # weight and points for current lGrid index
			
			for iLPC in range(iLPC_lower, iLPC_upper+1):
				# Is this a point below or above current lGrid point
				if lPointCloud[iLPC] < lGrid[iLG]:
					weight = 1-abs(lGrid[iLG] - lPointCloud[iLPC])/width_lower					
				elif lPointCloud[iLPC] > lGrid[iLG]:
					weight = 1-abs(lPointCloud[iLPC] - lGrid[iLG])/width_upper

				#Cheating !! this may be breaking something in odf calc
				if abs(weight)>1:
					weight = 0
					#print('iLG={} iLPC={} lG[iLG]={} lPC[iLPC]={} weight={}'.format(iLG, iLPC, lGrid[iLG], lPointCloud[iLPC], weight))
				iLG_WM.append([iLPC, weight])  # Save pointCloud index and associated lbin weight
			lMapWeights[iLG] = iLG_WM
		"""
		for iLG in range(Nl):
			print('iLG={}  L={} nP={}'.format(iLG, lGrid[iLG], len(lMapWeights[iLG])))
			for i in range(len(lMapWeights[iLG])):
				iPC = lMapWeights[iLG][i][0]
				print('  Lpc={} iPC={} weight={}'.format(lPointCloud[iPC], iPC, lMapWeights[iLG][i][1]))
		"""
		#Calc eGrid mapping weights
		eMapFracs = [ [ [] for e in range(Ne)] for pnt in range(Npc)] # Npc x Ne x (nx2)
		for iPC in range(Npc):
			i,j = iPointCloud[iPC]
			vm = vms[i,j]
			ke_center = alamData['ilamc']*vm
			ke_lower = alamData['ilami'][:-1]*vm
			ke_upper = alamData['ilami'][1:]*vm
			ke_width = ke_upper-ke_lower

			e_lower = np.zeros(Ne)
			e_upper = np.zeros(Ne)
			for iEG in range(Ne):
				e_lower[iEG] = eGrid[0] if iEG == 0 else (eGrid[iEG-1] + eGrid[iEG])/2
				e_upper[iEG] = eGrid[-1] if iEG == Ne-1 else (eGrid[iEG+1] + eGrid[iEG])/2

			eMapFracs[iPC] = scutils.getWeights_ConsArea(ke_center, ke_lower, ke_upper, eGrid, e_lower, e_upper)

			# Iterate over eGrid, grab weights for all overlapping k bins
			"""
			for iEG in range(Ne):
				e_lower = eGrid[0] if iEG == 0 else (eGrid[iEG-1] + eGrid[iEG])/2
				e_upper = eGrid[-1] if iEG == Ne-1 else (eGrid[iEG+1] + eGrid[iEG])/2

				frac_arr = []
				#Loop through k values, find k bind in 1D collision with current eGrid element
				for k in range(Nk):
					#Does it collide
					if e_lower < ke_upper[k] and e_upper > ke_lower[k]:
						#Get overlap bounds
						o_lower = ke_lower[k] if ke_lower[k] > e_lower else e_lower
						o_upper = ke_upper[k] if ke_upper[k] < e_upper else e_upper
						o_width = o_upper-o_lower
						#frac_arr.append([k, eetas[k,i,j]*o_width/ke_width])  # Fraction of k bin that coincides with eGrid bin
						frac_arr.append([k, o_width/ke_width[k]])  # Fraction of k bin that coincides with eGrid bin
				#print('n: {}  arr:{}'.format(len(frac_arr), frac_arr))
				eMapFracs[iPC][iEG] = frac_arr
			"""
		"""
		for iPC in range(Npc):
			print('iPC={}  L={}'.format(iPC, lPointCloud[iPC]))
			i,j = iPointCloud[iPC]
			vm = vms[i,j]
			for e in range(Ne):
				print('  e={}'.format(eGrid[e]))
				for i in range(len(eMapFracs[iPC][e])):
					k = eMapFracs[iPC][e][i][0]
					print('    k={}  E_k={} frac={}'.format(k, alamData['ilamc'][k]*vm,eMapFracs[iPC][e][i][1]))
		"""

		#Calculate the desired variables for all of our points
		odf_pc = np.zeros((Npc,Nk))
		press_pc = np.zeros((Npc,Nk))
		for npc in range(Npc):
			# Single point's values
			i,j = iPointCloud[npc]
			vm = vms[i,j]
			eeta_k = eetas[:,i,j]

			odf_pc[npc,:] =  sf_factor*alamData['ilamc']*vm*eeta_k/alamData['lamscl'] # [Nk]
			press_pc[npc,:] =  pressure_factor*alamData['ilamc']*eeta_k*vm**2.5 * 1E9  # [Pa -> nPa]
			#print('nPC={} vm={} odf={} press={}'.format(npc,vm,odf_pc[npc,:],press_pc[npc,:]))
			
		#Now we have all necessary info. Map all points to their corresponding L-E grid point
		odf_EL = np.zeros((Ne,Nl))
		press_EL = np.zeros((Ne,Nl))
		for iLG in range(Nl):
			#print('iLG=' + str(iLG))
			pointsToMap = lMapWeights[iLG]  # [nx2], nx[pc index, weight]
			numPoints = len(pointsToMap)

			for i in range(numPoints):
				iPC = pointsToMap[i][0]
				lWeight = pointsToMap[i][1]
				
				for e in range(Ne):
					for ik in range(len(eMapFracs[iPC][e])):
						k = eMapFracs[iPC][e][ik][0]
						eFrac = eMapFracs[iPC][e][ik][1]
						odf_EL[e,iLG] += lWeight/numPoints * eFrac * odf_pc[iPC,k]
						press_EL[e,iLG] += lWeight/numPoints * eFrac * press_pc[iPC,k]

		#Yay we made it
		odf_tkl[n,:,:] = odf_EL
		press_tkl[n,:,:] = press_EL

	#Destroy our hard work
	odf_tl = np.ma.sum(np.ma.masked_invalid(odf_tkl), axis=1)
	press_tl = np.ma.sum(np.ma.masked_invalid(press_tkl), axis=1)
		
	result = {}
	result['T'         ] = rcmTimes['T'  ][iTStart:iTEnd+1:sStride]
	result['MJD'       ] = rcmTimes['MJD'][iTStart:iTEnd+1:sStride]
	result['lambda'    ] = alamData['ilamc']
	result['L_bins'    ] = lGrid
	result['energyGrid'] = eGrid
	result['odf_tkl'   ] = odf_tkl
	result['odf_tl'    ] = odf_tl
	result['press_tkl' ] = press_tkl
	result['press_tl'  ] = press_tl

	if dojson: kj.dump(rcmjfname, result)

	return result

#TODO: Take list of variable strings to pull from rcm.h5 file
def getRCM_eqlatlon(mhdrcmf5, rcmTimes, sStart, sEnd, sStride, jdir=None, forceCalc=False):
	"""Grab certain variables along with equatorial and lat-lon grid.

	Args:
		rcmf5 (str): RCM HDF5 filename.
		rcmTimes (dict): Dictionary returned from getRCMTimes().
		evMJDs (float): MJD to evaluate cumulative fractions.
		evalLatLons (list): [lat, lon] location to perform calculation.
		species (str, optional): Species to calculate cumulative fractions for. Defaults to 'ions'.
		jdir (str, optional): Directory to find JSON files in. Defaults to None.
		forceCalc (bool, optional): Force calculation even if JSON file exists. Defaults to False.

	Returns:
		dict: A dictionary containing the calculated results.

	Raises:
		None
	"""
	if jdir is not None:
		dojson = True
		rcmjfname = os.path.join(jdir, RCM_EQLATLON_JFNAME)
	else:
		dojson = False

	if dojson and not forceCalc:
		if os.path.exists(rcmjfname):
			print("Grabbing RCM eq_lat-lon data from " + rcmjfname)
			return kj.load(rcmjfname)

	Nt = rcmTimes['Nt']
	iTStart = np.abs(rcmTimes['sIDs']-sStart).argmin()
	iTEnd = np.abs(rcmTimes['sIDs']-sEnd).argmin()
	sIDs = rcmTimes['sIDs'][iTStart:iTEnd+1:sStride]
	sIDstrs = rcmTimes['sIDstrs'][iTStart:iTEnd+1:sStride]

	mhdrcm5 = h5.File(mhdrcmf5,'r')
	
	mhdrcmS0 = mhdrcm5[sIDstrs[0]]
	Ni,Nj = mhdrcmS0['aloct'].shape

	mlat_rcm = 90.0-mhdrcmS0['colat'][0,:]*180/np.pi
	mlon_rcm = mhdrcmS0['aloct'][:,0]*180/np.pi
	mlatrcm_min = np.min(mlat_rcm)
	mlatrcm_max = np.max(mlat_rcm)

	#Get desired variables for all timesteps
	xmin_arr = np.ma.zeros((Nt, Ni, Nj))
	ymin_arr = np.ma.zeros((Nt, Ni, Nj))
	press_arr = np.ma.zeros((Nt, Ni, Nj))
	dens_arr = np.zeros((Nt, Ni, Nj))
	
	scLoc_eq = np.zeros((Nt, 2))
	scLoc_latlon = np.zeros((Nt, 2))

	rMin = 1.25
	rMax = 35.0
	ioCut = -0.5
	pCut = 1E-8
	
	print("Grabbing data...")
	"""
	for t in range(len(sIDstrs)):
		xm = mhdrcm5[sIDstrs[t]]['xMin'][:]
		ym = mhdrcm5[sIDstrs[t]]['yMin'][:]
		bmR = np.sqrt(xm*xm + ym*ym)
		pm = mhdrcm5[sIDstrs[t]]['P'][:]
		#Turn IOpen into a big true/false map
		iopen_t = mhdrcm5[sIDstrs[t]]['IOpen'][:] 

		Ir = (bmR < rMin) | (bmR > rMax)
		I_m = Ir | (iopen_t > ioCut) | (pm < pCut)
	"""
	#import kaipy.gamera.msphViz as mviz
	rcmdata = gampp.GameraPipe('',mhdrcmf5.split('.h5')[0])
	for t in range(0, len(sIDs)):
		bmX, bmY = rcmpp.RCMEq(rcmdata, sIDs[t], doMask=True)
		I = rcmpp.GetMask(rcmdata, sIDs[t])
		pm = rcmpp.GetVarMask(rcmdata, sIDs[t], 'P', I)

		xmin_arr[t,:,:] = np.transpose(bmX)
		ymin_arr[t,:,:] = np.transpose(bmY)
		press_arr[t,:,:] = np.transpose(pm)

	"""
	for t in range(tStart, len(sIDstrs)):
		#Linterp sc location based on time
		rcmTime = rcm_ut[t]
		if ut[0] > rcmTime or ut[-1] < rcmTime:
			scLoc_eq[t,:] = [0,0]
			scLoc_latlon[t,:] = [0,0]
		else:
			itime = 0
			while ut[itime] < rcmTime: itime += 1

			scLoc_eq[t,:] = [xeq_sc[itime], yeq_sc[itime]]
			scLoc_latlon[t,:] = [mlat_sc[itime], mlon_sc[itime]]
	"""
	result = {}
	result['T']     = rcmTimes['T'][iTStart:iTEnd+1:sStride]
	result['MJD']   = rcmTimes['MJD'][iTStart:iTEnd+1:sStride]
	result['MLAT']  = mlat_rcm
	result['MLON']  = mlon_rcm
	result['xmin']  = xmin_arr
	result['ymin']  = ymin_arr
	result['press'] = press_arr

	if dojson: kj.dump(rcmjfname, result)

	return result

def getRCM_CumulFrac(rcmf5, rcmTimes, evMJD, evalLatLon, species='ions', jdir=None,forceCalc=False):
	"""Calculate cumulative fractions of certain vars (ex. pressure) over energy channels, for each list of points for each given MJD.

	Args:
		rcmf5 (str): RCM HDF5 filename.
		rcmTimes (dict): Dictionary returned from getRCMTimes().
		evMJDs (list): List of MJDs to evaluate cumulative fractions.
		evalLatLons (list): List of [lat, lon] locations to perform calculation.

	Returns:
		dict: Result data structure containing the following keys:
			- 'MJD': MJD value.
			- 'i': i value.
			- 'j': j value.
			- 'lat': Latitude value.
			- 'lon': Longitude value.
			- 'xmin': xmin value.
			- 'ymin': ymin value.
			- 'vm': vm value.
			- 'energies': List of rcm energies [eV].
			- 'Ptot': Total pressure.
			- 'Ppar': List of each channel's individual pressure.
			- 'Pcum': List of each channel's cumulative pressure (summing from k=0).
	"""

	if jdir is not None:
		dojson = True
		rcmjfname = os.path.join(jdir, RCM_CUMULFRAC_JFNAME)
	else:
		dojson = False

	if dojson and not forceCalc:
		if os.path.exists(rcmjfname):
			print("Looking for RCM cumulative fraction data from " + rcmjfname)
			#In this case, saved variable is a list with all points saved so far in one big heap
			data = kj.load(rcmjfname)
			for point in data:
				if point['MJD'] == evMJD and point['lat'] == evalLatLon[0] and point['lon'] == evalLatLon[1]:
					return point

			print("Point not found in previous data, calculating")

	#Unpack RCM info
	sIDs = rcmTimes['sIDs']
	sIDstrs = rcmTimes['sIDstrs']
	rcmMJDs = rcmTimes['MJD']
	rcm5 = h5.File(rcmf5,'r')
	rcmS0 = rcm5[sIDstrs[0]]

	rcmLats = 90 - rcmS0['colat'][0,:]*180/np.pi
	rcmLons = rcmS0['aloct'][2:,0]*180/np.pi

	evLat = evalLatLon[0]
	evLon = evalLatLon[1]

	alamData = getSpecieslambdata(rcmS0, species)
	Nk = len(alamData['ilamc'])

	#Variable we're gonna return
	result = {}

	#TODO: Currently being lazy and getting the closest RCM point. Should implement TSC
	#Get RCM data closest to desired MJD, lat and lon
	ircmStep = np.abs(rcmMJDs-evMJD).argmin()
	i = np.abs(rcmLats-evLat).argmin()
	j = np.abs(rcmLons-evLon).argmin()
	stepStr = sIDstrs[ircmStep]
	rcmS5 = rcm5[stepStr]

	point = {}
	point['MJD'] = evMJD
	point['i'] = i
	point['j'] = j
	point['lat'] = rcmLats[i]
	point['lon'] = rcmLons[j]
	point['xmin'] = rcmS5['rcmxmin'][j,i]
	point['ymin'] = rcmS5['rcmymin'][j,i]
	vm = rcmS5['rcmvm'][j,i]
	point['vm'] = rcmS5['rcmvm'][j,i]
	point['energies'] = alamData['ilamc']*vm  # [eV]

	eetas = rcmS5['rcmeeta'][alamData['kStart']:alamData['kEnd'],j,i]

	pTot = 0
	pPar = np.zeros(Nk)
	pCum = np.zeros(Nk)

	pPar = pressure_factor*alamData['ilamc']*eetas*vm**2.5 * 1E9  # [Pa -> nPa], partial pressures for each channel
	pCum[0] = pPar[0] # First bin's partial press = cumulative pressure
	for k in range(1,Nk):
		pCum[k] = pCum[k-1] + pPar[k]  # Get current cumulative pressure by adding this bin's partial onto last bin's cumulative
	pTot = pCum[-1] # Last bin's cumulative = total pressure

	point['Ptot'] = pTot
	point['Ppar'] = pPar
	point['Pcum'] = pCum

	if dojson: 
		if os.path.exists(rcmjfname):
			data = kj.load(rcmjfname)
			for i in len(range(data)):
				#Remove old point if it exists
				pnt = data[i]
				if pnt['MJD'] == evMJD and pnt['lat'] == evalLatLon[0] and pnt['lon'] == evalLatLon[1]:
					data.remove(i)
				data.append(point)

			kj.dump(rcmjfname, data)
		else:
			data = [point]
			kj.dump(rcmjfname, data)

	return point
#======
#Plotting
#======

def plt_ODF_Comp(AxSC, AxRCM, AxCB, odfData, mjd=None, cmapName='CMRmap', norm=None, forcePop=False):
	"""
	Plot the ODF (Orientation Distribution Function) comparison between AxSC and AxRCM.

	Args:
		AxSC (matplotlib.axes.Axes): The subplot for the SC (Spacecraft) ODF.
		AxRCM (matplotlib.axes.Axes): The subplot for the RCM (Remote Component Manipulator) ODF.
		AxCB (matplotlib.axes.Axes): The colorbar axis.
		odfData (dict): A dictionary containing the ODF data.
		mjd (float): The Modified Julian Date (MJD) to highlight on the plot (optional).
		cmapName (str): The name of the colormap to use (default: 'CMRmap').
		norm (matplotlib.colors.Normalize): The normalization object for the colorbar (optional).
		forcePop (bool): A flag indicating whether to force population of the subplots (default: False).

	Returns:
		None
	"""
	axIsPopulated = not AxSC.get_ylabel() == ''

	eGrid   = odfData['energyGrid']
	scTime  = odfData['sc']['time']
	scODF   = odfData['sc']['diffFlux']
	rcmTime = odfData['rcm']['time']
	rcmODF  = odfData['rcm']['diffFlux']

	#ut = scutils.mjd_to_ut(rcmTime)
	ut = kT.MJD2UT(rcmTime)

	if norm is None:
		vMax = np.max([scODF.max(), rcmODF.max()])
		vMin = np.max([scODF[scODF>0].min(), rcmODF[rcmODF>0].min()])
		norm = kv.genNorm(vMin,vMax,doLog=True)

	if not axIsPopulated or forcePop:
		kv.genCB(AxCB,norm,r'Intensity [$cm^{-2} sr^{-1} s^{-1} keV^{-1}$]',cM=cmapName,doVert=True,cbSz=14)

		AxSC.pcolormesh(scTime, eGrid, np.transpose(scODF), norm=norm, shading='nearest', cmap=cmapName)

		AxSC.set_xlim([ut[0], ut[-1]])
		AxSC.set_ylabel("%s Energy [keV]"%odfData['sc']['name'])
		AxSC.set_yscale('log')
		#AxSC.xaxis.set_major_formatter(plt.NullFormatter())

		AxRCM.pcolormesh(ut, eGrid, np.transpose(rcmODF), norm=norm, shading='nearest', cmap=cmapName)
		#for n in range(len(rcmTime)):
		#    AxRCM.plot(odfData['rcm']['origEGrid'][n,:], odfData['rcm']['origODF'][n,:])
		AxRCM.set_ylabel("RCM Energy [keV]")
		AxRCM.set_yscale('log')

		#xlblFmt = mdates.DateFormatter('%H:%M')
		#AxRCM.xaxis.set_major_formatter(xlblFmt)

	if mjd is not None:
		if mjd < rcmTime[0] or mjd > rcmTime[-1]:
			print(str(mjd) + "not in rcm data, exiting")
			return
		iMJD = np.abs(rcmTime - mjd).argmin()
		lineUT = ut[iMJD]

		if len(AxSC.lines) != 0:
			AxSC.lines.pop(0)  #Remove previous mjd line
			AxSC.lines.pop(0)  #Remove previous mjd line
			AxRCM.lines.pop(0)
			AxRCM.lines.pop(0)
		yMin, yMax = AxSC.get_ylim()
		AxSC.plot([lineUT, lineUT], [yMin, yMax], '-k',linewidth=2)
		AxSC.plot([lineUT, lineUT], [yMin, yMax], '-w',linewidth=1)
		yMin, yMax = AxRCM.get_ylim()
		AxRCM.plot([lineUT, lineUT], [yMin, yMax], '-k',linewidth=2)
		AxRCM.plot([lineUT, lineUT], [yMin, yMax], '-w',linewidth=1)
		

def plt_tl(AxTL, tkldata, AxCB=None, mjd=None,cmapName='CMRmap',norm=None):
	"""
	Plot the total pressure as a function of L shell and time.

	Args:
		AxTL: The matplotlib Axes object for the L vs. Time plot.
		tkldata: A dictionary containing the data for the plot.
		AxCB: The matplotlib Axes object for the colorbar plot (optional).
		mjd: The Modified Julian Date for a specific time point (optional).
		cmapName: The name of the colormap to use (default: 'CMRmap').
		norm: The normalization object for the color mapping (optional).

	Returns:
		None
	"""

	L_arr = tkldata['L_bins']
	ut = kT.MJD2UT(tkldata['MJD'])
	press_tl = np.array(tkldata['press_tl'][:], dtype=float)  # Need to do this to handle masked stuff
	press_tl = np.ma.masked_invalid(press_tl)

	#Initialize static plots if hasn't been done yet
	doPopulateTL =  AxTL.get_ylabel() == ''
	doPopulateCB = AxCB is not None and AxCB.get_label() == ''

	if norm is None: 
		norm = genVarNorm(press_tl, doLog=True)

	if doPopulateTL:
		#L vs. Time
		AxTL.pcolormesh(ut, L_arr, np.transpose(press_tl), norm=norm, shading='nearest', cmap=cmapName)
		AxTL.set_xlim([ut[0], ut[-1]])
		AxTL.set_ylabel('L shell')
		AxTL.set_xlabel('UT')
		#Sim boundaries
		xmin,xmax = AxTL.get_xlim()
		AxTL.plot([xmin,xmax],[-tkl_lInner,-tkl_lInner], 'w--')
		AxTL.plot([xmin,xmax],[tkl_lInner,tkl_lInner], 'w--')
	if doPopulateCB:
		kv.genCB(AxCB, norm, r'Total pressure [$nPa$]', cM=cmapName, doVert=False)

	#Time-specific stuff
	if mjd is not None:
		if mjd < tkldata['MJD'][0] or mjd > tkldata['MJD'][-1]:
			return
		iMJD = np.abs(tkldata['MJD'] - mjd).argmin()
		lineUT = ut[iMJD]

		if len(AxTL.lines) > 2:
			AxTL.lines.pop(2)  #Remove previous mjd line
		yMin, yMax = AxTL.get_ylim()
		AxTL.plot([lineUT, lineUT], [yMin, yMax], '-k')


def plt_rcm_eqlatlon(AxLatlon, AxEq, rcmData, satTrackData=None, AxCB=None, mjd=None, norm=None, cmapName='viridis'):
	"""
	Plot RCM data on equal latitude-longitude and equatorial coordinate systems.

	Args:
		AxLatlon (matplotlib.axes.Axes): The axis object for the equal latitude-longitude plot.
		AxEq (matplotlib.axes.Axes): The axis object for the equatorial coordinate plot.
		rcmData (dict): A dictionary containing RCM data.
		satTrackData (dict, optional): A dictionary containing satellite track data. Default is None.
		AxCB (matplotlib.axes.Axes, optional): The colorbar axis object. Default is None.
		mjd (float, optional): The Modified Julian Date. Default is None.
		norm (matplotlib.colors.Normalize, optional): The normalization object for the color mapping. Default is None.
		cmapName (str, optional): The name of the colormap. Default is 'viridis'.

	Returns:
		None
	"""

	mjd_arr   = rcmData['MJD']
	xmin_arr  = rcmData['xmin']
	ymin_arr  = rcmData['ymin']
	mlat_arr  = rcmData['MLAT']
	mlon_arr  = rcmData['MLON']
	press_arr = rcmData['press']

	#ut = scutils.mjd_to_ut(rcmData['MJD'])
	ut = kT.MJD2UT(rcmData['MJD'])
	Nt = len(ut)

	if norm is None: 
		norm = genVarNorm(press_arr, doLog=True)

	#Initialize static plots if hasn't been done yet
	if AxCB is not None:
		AxCB = kv.genCB(AxCB, norm, r'Pressure [$nPa$]', cM=cmapName, doVert=False)

	if mjd is not None:
		if mjd < mjd_arr[0] or mjd > mjd_arr[-1]:
			print(str(mjd) + "not in rcm data, exiting")
			return
		iMJD = np.abs(mjd_arr - mjd).argmin()
		lineUT = ut[iMJD]

		AxLatlon.clear()
		AxEq.clear()

		#Prep rcm lat/lons for polar plotting
		riono = np.cos(mlat_arr*np.pi/180.)
		tiono = np.concatenate((mlon_arr, [mlon_arr[0]]))*np.pi/180.
		AxLatlon.pcolor(tiono, riono, to_center(np.transpose(press_arr[iMJD])),norm=norm, shading='auto', cmap=cmapName)
		AxLatlon.axis([0, 2*np.pi, 0, 0.7])

		AxEq.pcolor(xmin_arr[iMJD], ymin_arr[iMJD], to_center(press_arr[iMJD]), norm=norm, shading='auto', cmap=cmapName)

		#Draw satellite location
		if satTrackData is not None:
			x_sc = satTrackData['xeq']
			y_sc = satTrackData['yeq']
			req_sc = satTrackData['Req']
			iscMJD = np.abs(satTrackData['MJD'] - mjd).argmin()
			if req_sc[iscMJD] > 1E-8:
				leadMax = iscMJD
				while leadMax < min(iscMJD+80, Nt) and req_sc[leadMax] > 1E-8: leadMax += 1 #!!This isn't working as intended for some reason
				AxEq.plot(x_sc[iscMJD:leadMax], y_sc[iscMJD:leadMax], 'k-')
				
				satCircle = plt.Circle((x_sc[iscMJD], y_sc[iscMJD]), 0.15, color='black')
				AxEq.add_patch(satCircle)
		kv.addEarth2D(ax=AxEq)

		#Set bounds
		boxRatio = get_aspect(AxEq)
		xbMin = -10
		xbMax = 10
		dx = xbMax - xbMin
		dy = boxRatio*dx
		ybMin = -dy/2
		ybMax = dy/2
		AxEq.set_xlim([xbMin, xbMax])
		AxEq.set_ylim([ybMin, ybMax])







