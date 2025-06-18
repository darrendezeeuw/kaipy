#Various routines to generate RCM data
# Standard modules
import os
import importlib.resources as pkg_resources

# Third-party modules
import numpy as np

def LoadLAS1(fIn="rcmlas1"):
	"""
	Load LAS1 data from a file.

	Args:
		fIn (str): The filename of the LAS1 data file. Default is "rcmlas1".

	Returns:
		alamc (ndarray): Array of alamc values.
		etac (ndarray): Array of etac values.
		ikflavc (ndarray): Array of ikflavc values.
		fudgec (ndarray): Array of fudgec values.
	"""


	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	fIn = os.path.join(__location__, fIn)

	print("Reading %s" % (fIn))
	Q = np.loadtxt(fIn, skiprows=2)

	alamc = Q[:, 0]
	etac = Q[:, 1]
	ikflavc = Q[:, 2]
	fudgec = Q[:, 3]

	return alamc, etac, ikflavc, fudgec


def LoadDKTab(fIn="dktable"):
	"""
	Load the DK table from a file and return a flattened array.

	Args:
		fIn (str): The name of the file containing the DK table. Default is 'dktable'.

	Returns:
		numpy.ndarray: A flattened array representing the DK table.
	"""
	from kaipy import rcm
	#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	fIn = pkg_resources.files(rcm).joinpath(fIn)

	print("Reading %s" % (fIn))

	Q = np.loadtxt(fIn)

	return Q.flatten()


def LoadEnchan(fIn="enchan.dat"):
	"""
	LoadEnchan function loads data from a file and returns two arrays.

	Args:
		fIn (str): The name of the input file. Default is "enchan.dat".

	Returns:
		iflavin (numpy.ndarray): Array containing the values from the first column of the input file.
		alamin (numpy.ndarray): Array containing the values from the second column of the input file.
	"""
	import os
	import numpy as np
	from kaipy import rcm
	#__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

	fIn = pkg_resources.files(rcm).joinpath(fIn)

	print("Reading %s" % (fIn))

	Q = np.loadtxt(fIn, skiprows=1)

	iflavin = Q[:, 0]
	alamin = Q[:, 1]

	return iflavin, alamin		