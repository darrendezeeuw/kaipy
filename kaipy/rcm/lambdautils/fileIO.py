import os
import numpy as np
import json
import h5py as h5
from dataclasses import asdict as dc_asdict

import kaipy.kaijson as kj
import kaipy.rcm.lambdautils.AlamParams as aP
import kaipy.rcm.lambdautils.DistTypes as dT

try:
	import kaipy.rcm.rcminit as rcminit
	hasKaiRCM = True
except:
	print("Couldn't load kaipy.rcm.init, can't add dktable to rcmconfig file.")
	hasKaiRCM = False


def saveRCMConfig(alamData, params=None, fname='rcmconfig.h5'):
	"""
	Save the RCM configuration data to an HDF5 file.

	Args:
		alamData: The RCM configuration data to be saved.
		params: Additional parameters to be saved (optional).
		fname: The filename of the HDF5 file (default: 'rcmconfig.h5').
	"""
	f5 = h5.File(fname, 'w')
	saveData(f5, alamData)
	if params is not None:
		saveParams(f5, params)
	f5.close()


#These save and load AlamParams so that we can always tell how a lambda distribution was generated
def saveParams(f5, alamParams):
	"""
	Save the alamParams to the given f5 object.

	Args:
		f5: The f5 object to save the parameters to.
		alamParams: The alamParams to be saved.

	Returns:
		None
	"""
	f5.attrs['AlamParams'] = kj.dumps(dc_asdict(alamParams), noIndent=True)


def loadParams(f5):
	"""
	Load parameters from a file.

	Args:
		f5: The file object containing the parameters.

	Returns:
		aPObj: The loaded parameters as an AlamParams object.

	Raises:
		AttributeError: If the dictionary cannot be converted into an object.

	Notes:
		- This function assumes that the file object `f5` has an attribute 'AlamParams' which contains a JSON string.
		- If the 'AlamParams' attribute cannot be converted into an object, the function will return the parameters as a dictionary.
		- The 'distType' attribute of each 'specParams' object is instantiated separately due to the use of inherited classes.
	"""
	aPDict = kj.loads(f5.attrs['AlamParams'])
	try:
		aPObj = aP.AlamParams.from_dict(aPDict)
	except AttributeError: 
		print("ERROR (loadParams): Can't turn dictionary into object. Please install the dataclasses_json module")
		print("                     Returning as dictionary instead")
		return aPDict
	# DistTypes won't load correctly using from_dict because we're using inherited classes, need to instantiate ourselves
	for sPDict, sPObj in zip(aPDict['specParams'], aPObj.specParams):
		sPObj.distType = dT.getDistTypeFromKwargs(**sPDict['distType'])
	
	return aPObj


def saveData(f5, alamData, doPrint=False):
	"""
	Takes an AlamData object, formats it to rcmconfig.h5 style, and saves it.

	Parameters:
		f5 (h5py.File): The HDF5 file object to save the data to.
		alamData (AlamData): The AlamData object containing the data to be saved.
		doPrint (bool, optional): Whether to print the intermediate arrays. Defaults to False.
	"""
	lambdas = np.array([])
	flavs = np.array([], dtype=int)
	fudges = np.array([])

	for spec in alamData.specs:
		lambdas = np.append(lambdas, spec.alams)
		flavs = np.append(flavs, [spec.flav for f in range(spec.n)])
		fudges = np.append(fudges, [spec.fudge for f in range(spec.n)])

	if doPrint:
		print(lambdas)
		print(flavs)
		print(fudges)

	f5.create_dataset('alamc', data=lambdas)
	f5.create_dataset('ikflavc', data=flavs)
	f5.create_dataset('fudgec', data=fudges)

	if hasKaiRCM:
		dktab = rcminit.LoadDKTab()
		f5.create_dataset('dktable', data=dktab)
	else:
		print("Couldn't load kaipy.rcm.init, can't add dktable to rcmconfig file.")

