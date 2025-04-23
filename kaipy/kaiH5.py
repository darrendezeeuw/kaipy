import h5py
import numpy as np
import os, sys, subprocess
from alive_progress import alive_bar, alive_it
import datetime
from astropy.time import Time
#Import kaipy modules
import kaipy.kdefs as kdefs
import kaipy.kaiTools as ktools


#------
# Data Containers
#------

class H5Info(object):
	""" 
	Class to hold model data information from h5 output files

	Args:
		h5fname (str): .h5 filename to scrape info from
		noSubsec (bool): Whether or not UTs should include subseconds (optional, default: True)

	Attributes:
		fname (str): .h5 filename the rest of the info came from
		steps (List[int]): list of step numbers
		stepStrs (List[str]): list of step strings in Step#X h5.Group format
		Nt (int): Number of time steps
		times (List[float]): List of times corresponding to each step
			Units: Unchanged from file, likely seconds
		MJDs (List[float]): List of MJDs corresponding to each step
		UTs (List[float]): List of UTs corresponding to each step
			May or may not include subseconds based on constructor args
	"""

	def __init__(self, h5fname: str, noSubsec:bool=True):
		# h5fname = h5fname.split('/')[-1]
		self.fname = h5fname
		self.Nt, self.steps = cntSteps(self.fname)
		print("Found {} steps in {}".format(self.Nt, self.fname))
		self.stepStrs = ['Step#'+str(s) for s in self.steps]
		self.times = getTs(self.fname, self.steps, "time")
		self.MJDs  = getTs(self.fname, self.steps, "MJD" )

		f5 = h5py.File(h5fname)
		if noSubsec:
			ut_arr = np.zeros(self.Nt, dtype=datetime.datetime)
			for i,mjd in enumerate(self.MJDs):
				UTStr = Time(mjd,format='mjd').isot
				utTrim = UTStr.split('.')[0]
				UT = datetime.datetime.strptime(utTrim,'%Y-%m-%dT%H:%M:%S')
				ut_arr[i] = UT
		else:
			ut_arr = ktools.MJD2UT(self.MJDs)

		self.UTs = ut_arr

		f5.close()

	def printStepInfo(self) -> None:
		"""
		Prints a summary of step info for the contained data

		Args: None

		Returns: None
		"""
		print("Step Info for {}:".format(self.fname))
		print("  Step start/end/stride : {} / {} / {}".format(self.steps[0], self.steps[-1], self.steps[1]-self.steps[0]))
		print("  Time start/end/stride : {:1.2f} / {:1.2f} / {:1.2f}".format(self.times[0], self.times[-1], self.times[1]-self.times[0]))
		print("  MJD  start/end        : {:1.8f} / {:1.8f}".format(self.MJDs[0], self.MJDs[-1]))
		print("  UT   start/end        : {} / {}".format(self.UTs[0], self.UTs[-1]))


class TPInfo(H5Info):
	"""
	Extension of H5Info for test particle files

	Attributes:
		Ntp (int): Number of test particles
		id2idxMap (Dict{int:}): Dict to help map TP id to its index
	"""

	def __init__(self, fname:str, noSubsec:bool=True):
		super().__init__(fname, noSubsec)
		with h5py.File(self.fname) as tp5:
			ids = tp5[self.stepStrs[0]]['id']
			self.Ntp = ids.shape[0]
			self.id2idxMap = {}
			for i in range(self.Ntp):
				self.id2idxMap[int(ids[i])] = int(i)

#------
# General functions
#------

#Generate MPI-style name
def genName(bStr, i, j, k, Ri, Rj, Rk, nRes=None):
	'''
	Generate a filename based on the given parameters.

	Args:
		bStr (str): The base string for the filename.
		i (int): The value of i.
		j (int): The value of j.
		k (int): The value of k.
		Ri (int): The value of Ri.
		Rj (int): The value of Rj.
		Rk (int): The value of Rk.
		nRes (int, optional): The value of nRes. Defaults to None.

	Returns:
		str: The generated filename.
	'''

	n = k + j * Rk + i * Rj * Rk
	if nRes is None:
		fID = bStr + "_%04d_%04d_%04d_%04d_%04d_%04d.gam.h5" % (Ri, Rj, Rk, i, j, k)
	else:
		fID = bStr + "_%04d_%04d_%04d_%04d_%04d_%04d" % (Ri, Rj, Rk, i, j, k) + ".gam.Res.%05d.h5" % (nRes)
	return fID


#Generate old-style MPI name
def genNameOld(bStr, i, j, k, Ri, Rj, Rk, nRes=None):
	'''
	Generate a file name based on the given parameters.

	Args:
		bStr (str): Base string for the file name.
		i (int): Value of i for this segment.
		j (int): Value of j for this segement.
		k (int): Value of k for this segement.
		Ri (int): Value of Ri Ri for the overall decomposition.
		Rj (int): Value of Rj Ri for the overall decomposition.
		Rk (int): Value of Rk Ri for the overall decomposition.
		nRes (int, optional): Value of nRes for restart. Defaults to None.

	Returns:
		str: Generated file name.
	'''
	n = k + j * Rk + i * Rj * Rk
	if nRes is None:
		fID = bStr + "_%04d_%04d_%04d_%04d_%04d_%04d_%012d.h5" % (Ri, Rj, Rk, i, j, k, n)
	else:
		fID = bStr + "_%04d_%04d_%04d_%04d_%04d_%04d_%012d" % (Ri, Rj, Rk, i, j, k, n) + ".Res.%05d.h5" % (nRes)
	return fID


#Quick check and exit routine
def CheckOrDie(fname):
	'''
	Check if a file exists, and exit the program if it doesn't.

	Args:
		fname (str): The file name or path to check.

	Raises:
		SystemExit: If the file doesn't exist, the program will exit with an error message.

	'''
	import os.path
	isExist = os.path.exists(fname)
	if not isExist:
		if len(fname) == 0:
			fname = "<EMPTY>"
		sys.exit("Unable to find file: %s" % (fname))

#Check directory exists and make it if not
def CheckDirOrMake(fdir):
	'''
	Check if the given directory exists. If not, create the directory.

	Args:
		fdir (str): The directory path to check or create.

	Returns:
		None
	'''
	import os

	isDir = os.path.isdir(fdir)
	if (not isDir):
		print("Creating %s"%(fdir))
		os.makedirs(fdir)

# Stamp file with git hash (using this script's location)
def StampHash(fname):
	'''
	Adds the git hash of the current commit to the given HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.

	Returns:
		None
	'''
	with h5py.File(fname, 'a') as f5:
		cwd = os.path.dirname(os.path.realpath(__file__))
		try:
			gh = subprocess.check_output(['git', '-C', cwd, 'rev-parse', 'HEAD']).decode('ascii').strip()
		except:
			print("ERROR: Couldn't grab git hash")
			gh = "XXXXX"
		f5.attrs['GITHASH'] = gh

# Stamp file with git branch (using this script's location)
def StampBranch(fname):
	'''
	Adds the current Git branch name as an attribute to an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.

	Returns:
		None
	'''
	with h5py.File(fname, 'a') as f5:
		cwd = os.path.dirname(os.path.realpath(__file__))
		try:
			gb = subprocess.check_output(['git', '-C', cwd, 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
		except:
			print("ERROR: Couldn't grab git branch")
			gb = "XXXXX"
		f5.attrs['GITBRANCH'] = gb

#Get git hash from file if it exists
def GetHash(fname):
	'''
	Retrieves the GITHASH attribute value from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.

	Returns:
		str: The value of the GITHASH attribute as a string.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
	'''
	CheckOrDie(fname)
	with h5py.File(fname,'r') as hf:
		rStr = hf.attrs.get("GITHASH","NONE")
	try:
		hStr = rStr.decode('utf-8')
	except (UnicodeDecodeError, AttributeError):
		hStr = rStr
	return hStr

#Get git branch from file if it exists
def GetBranch(fname):
	'''
	Retrieves the Git branch associated with the given file.

	Args:
		fname (str): The path to the file.

	Returns:
		str: The Git branch name.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
	'''
	CheckOrDie(fname)
	with h5py.File(fname,'r') as hf:
		rStr = hf.attrs.get("GITBRANCH","NONE")
	try:
		hStr = rStr.decode('utf-8')
	except (UnicodeDecodeError, AttributeError):
		hStr = rStr
	return hStr

#Return time at step n
def tStep(fname, nStp=0, aID="time", aDef=0.0):
	'''
	Retrieve the value of a specific attribute from a given HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		nStp (int): The step number.
		aID (str): The attribute ID to retrieve.  Defaults to "time".
		aDef (float): The default value to return if the attribute is not found.

	Returns:
		float: The value of the specified attribute.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
	'''
	CheckOrDie(fname)
	with h5py.File(fname, 'r') as hf:
		gID = "Step#%d" % (nStp)
		if aID in hf[gID].attrs:
			t = hf[gID].attrs[aID]
		else:
			t = aDef
	return t

	
def cntSteps(fname, doTryRecover=True, s0=0):
	'''
	Count the number of steps and retrieve the step IDs from an h5 file.

	Args:
		fname (str): The path to the h5 file.
		doTryRecover (bool): Whether to try recovering unreadable steps. Default is True.
		s0 (int): The starting step ID when recovering unreadable steps. Default is 0.

	Returns:
		nSteps (int): The total number of steps.
		sIds (ndarray): An array of step IDs.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
		ValueError: If the steps are unreadable and doTryRecover is False.
	'''


	try:
		CheckOrDie(fname)
		with h5py.File(fname, 'r') as hf:
			if kdefs.grpTimeCache in hf.keys() and 'step' in hf[kdefs.grpTimeCache].keys():
				sIds = np.asarray(hf[kdefs.grpTimeCache]['step'])
				nSteps = sIds.size
			else:
				Steps = [grp for grp in alive_it(hf.keys(), title="#-Steps".ljust(kdefs.barLab), length=kdefs.barLen, bar=kdefs.barDef) if "Step#" in grp]
				sIds = np.array([str.split(s, "#")[-1] for s in Steps], dtype=int)
				sIds.sort()
				nSteps = len(Steps)

		return nSteps, sIds
	except (ValueError, IndexError) as e:
		print("!!Warning: h5 file contains unreadable steps")

	if not doTryRecover:
		print("  Can try again with 'cntSteps(fname,doTryRecover=True)'")
		raise ValueError("Unreadable steps")
	else:
		print("Trying to read again while skipping bad values")
		with h5py.File(fname, 'r') as hf:
			badCounts = 0
			s = s0
			sIds = []
			while badCounts < 2000:
				try:
					sName = 'Step#' + str(s)
					tryReadGrp = hf[sName]
					# If still here, step is readable
					sIds.append(s)
					print("Read step " + str(s))
				except ValueError:
					badCounts += 1
					print("Bad count on step " + str(s))
				s += 1
			nSteps = len(sIds)
			sIds = np.array(sIds)
			sIds.sort()
		return nSteps, sIds

#More general version of cntSteps, useful for Step#X/Line#Y
def cntX(fname, gID=None, StrX="/Step#"):
	'''
	Count the number of steps and return sorted step IDs from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		gID (str, optional): The group ID within the HDF5 file. If None, all groups will be considered.
		StrX (str, optional): The string pattern to search for in the group names.

	Returns:
		nSteps (int): The number of steps found.
		sIds (numpy.ndarray): An array of sorted step IDs.

	Example usage:
		nSteps, sIds = cntX('/path/to/file.h5', gID='group1', StrX='/Step#')
	'''

	with h5py.File(fname, 'r') as hf:
		if gID is not None:
			grps = hf[gID].values()
		else:
			grps = hf.values()
		
		grpNames = [str(grp.name) for grp in grps]
		Steps = [stp for stp in grpNames if StrX in stp]
		nSteps = len(Steps)

		sIds = np.array([str.split(s, "#")[-1] for s in Steps], dtype=int)
		sIds.sort()
		
		return nSteps, sIds


def getTs(fname, sIds=None, aID="time", aDef=0.0):
	'''
	Retrieve time series data from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		sIds (list, optional): List of step IDs to retrieve time series data for. If None, all steps will be used.
		aID (str, optional): The attribute ID to retrieve. Default is "time".
		aDef (float, optional): The default value to use if the attribute is not found. Default is 0.0.

	Returns:
		numpy.ndarray: An array containing the time series data.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
		KeyError: If the specified attribute ID is not found in the HDF5 file.
	'''

	if sIds is None:
		nSteps, sIds = cntSteps(fname)
	Nt = len(sIds)
	T = np.zeros(Nt)
	CheckOrDie(fname)
	titStr = "Time series: %s" % (aID)

	with h5py.File(fname, 'r') as hf:
		if kdefs.grpTimeCache in hf.keys():
			if aID in hf[kdefs.grpTimeCache]:
				T = np.asarray(hf[kdefs.grpTimeCache][aID])
			else:
				T[:] = aDef
		else:
			with alive_bar(Nt, title=titStr.ljust(kdefs.barLab), length=kdefs.barLen, bar=kdefs.barDef) as bar:
				for idx, n in enumerate(sIds):
					gId = "/Step#%d" % (n)
					T[idx] = hf[gId].attrs.get(aID, aDef)
					bar()
	return T

#Used by MageStep to find closest datetime
def LocDT(items, pivot):
	'''
	Locates the index of the item in the given list that is closest to the specified pivot value.

	Args:
		items (list): A list of items to search through.
		pivot (datetime): The pivot value to compare against.

	Returns:
		int: The index of the item in the list that is closest to the pivot value.
	'''
	items_array = np.array(items)
	m0 = min(items_array, key=lambda x: abs(x - pivot))
	i0 = (items_array == m0).argmax()
	return i0

def MageStep(T0, inFile):
	'''
	Identify the step number closest to the given datetime.

	Args:
		T0: The datetime to compare against.
		inFile: The input file.

	Returns:
		nStp: The step number closest to the given datetime.
	'''

	# Identify which step number is closest to datetime
	MJDs = getTs(inFile, aID='MJD')

	# Convert to datetime
	UTs = np.array(ktools.MJD2UT(MJDs))

	# Find the index of the closest step number
	nStp = LocDT(UTs, T0)

	print(nStp, UTs[nStp], T0)
	return nStp


#Get shape/dimension of grid
def getDims(fname, vID="X", doFlip=True):
	'''
	Get the dimensions of a dataset in an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		vID (str, optional): The ID of the dataset to retrieve the dimensions from. Default is "X".
		doFlip (bool, optional): Whether to flip the dimensions. Default is True.

	Returns:
		numpy.ndarray: The dimensions of the dataset.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
	'''
	CheckOrDie(fname)
	with h5py.File(fname, 'r') as hf:
		Dims = hf["/"][vID].shape
	Ds = np.array(Dims, dtype=int)
	if doFlip:
		Ds = np.flip(Ds, axis=0)
	return Ds

#Get root variables
def getRootVars(fname):
	'''
	Retrieves the root variables from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.

	Returns:
		list: A list of root variable names, excluding coordinates.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
	'''
	CheckOrDie(fname)

	with h5py.File(fname, 'r') as hf:
		vIds = [str(k) for k in hf.keys() if "Step" not in str(k)]
	
	# Remove coordinates from list of root variables
	xyzS = ["X", "Y", "Z"]
	vIds = [v for v in vIds if v not in xyzS]

	return vIds

#Get variables in initial Step
def getVars(fname, smin):
	'''
	Retrieve variable IDs from an HDF5 file for a given step number.

	Args:
		fname (str): The path to the HDF5 file.
		smin (int): The step number.

	Returns:
		list: A list of variable IDs.

	Raises:
		FileNotFoundError: If the specified file does not exist.
	'''
	CheckOrDie(fname)
	with h5py.File(fname, 'r') as hf:
		gId = "/Step#%d" % (smin)
		stp0 = hf[gId]
		vIds = [str(k) for k in stp0.keys()]
	return vIds

#Get variable data
def PullVarLoc(fname, vID, s0=None, slice=(), loc=None):
	'''
	Pull variable data from HDF5 file and pass through (i, j, k) MPI rank loc

	Args:
		fname (str): The path to the HDF5 file.
		vID (str): The variable ID to pull data from.
		s0 (int or None): The starting index of the data to pull. Defaults to None.
		slice (tuple or None): The slice object to apply to the data. Defaults to ().
		loc (tuple or None): The (i, j, k) MPI rank location to pass the data through. Defaults to None.

	Returns:
		tuple: A tuple containing the variable data (V) and the MPI rank location (loc).
	'''
	V = PullVar(fname, vID, s0, slice)
	return V, loc

#Get variable data
def PullVar(fname, vID, s0=None, slice=()):
	'''
	Pull variable data from HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		vID (str): The variable ID to pull from the file.
		s0 (int, optional): The step number. If provided, the variable will be pulled from the specified step group. If not provided, the variable will be pulled from the root group.
		slice (tuple, optional): The slice to apply to the variable data. Defaults to an empty tuple, which means no slicing will be applied.

	Returns:
		numpy.ndarray: The pulled variable data.
	Raises:
	CheckOrDieError: If the file does not exist or is not readable.
	'''
	CheckOrDie(fname)
	with h5py.File(fname, 'r') as hf:
		if s0 is None:
			V = hf[vID][slice].T
		else:
			gId = "/Step#%d" % (s0)
			V = hf[gId][vID][slice].T
	return V

#Get attribute data from Step#s0 or root (s0=None)
def PullAtt(fname, vID, s0=None, a0=None):
	'''
	Retrieve an attribute from an HDF5 file.

	Args:
		fname (str): The path to the HDF5 file.
		vID (str): The name of the attribute to retrieve.
		s0 (int, optional): The step number. If provided, the attribute will be retrieved from the group corresponding to the step number.

	Returns:
		The value of the attribute.

	Raises:
		CheckOrDieError: If the file does not exist or is not readable.
		KeyError: If the specified attribute or group does not exist.
	'''
	CheckOrDie(fname)
	with h5py.File(fname, 'r') as hf:
		if s0 is None:
			hfA = hf.attrs
		else:
			gID = "/Step#%d" % (s0)
			hfA = hf[gID].attrs

		if (vID not in hfA.keys()):
			#Use default
			Q = a0
		else:
			Q = hfA[vID]

	return Q

